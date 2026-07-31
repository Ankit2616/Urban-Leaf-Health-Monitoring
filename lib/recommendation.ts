import { fetchCsv, r2Url } from "@/lib/r2";
import { getFeatures } from "@/lib/data";

export type PhaseKey = "pre_event" | "first_event" | "second_event" | "post_event";

type Bounds = [number, number, number, number];

type ZoneDefinition = {
  id: string;
  label: string;
  bounds: Bounds;
  type: "core" | "edge" | "buffer" | "half";
};

export type RecommendationZone = {
  id: string;
  label: string;
  bounds: Bounds;
  centroid: { lon: number; lat: number };
  areaKm2: number;
  category: "Preferred zone" | "Conditional" | "Avoid";
  suitabilityScore: number;
  environmentalRisk: number;
  evidenceScore: number;
  coreOverlapPct: number;
  centralityPct: number;
  peripheralAccessPct: number;
  distanceFromCorePct: number;
  rationale: string;
  facts: string[];
};

export type RecommendationContext = {
  ndviDropPct: number;
  bsiRisePct: number;
  heatRisePct: number;
  disturbanceWindow: string;
  recoverySignal: string;
};

export type RecommendationData = {
  generatedAt: string;
  studyArea: {
    label: string;
    bounds: Bounds;
  };
  methodology: Array<{ label: string; weight: number; description: string }>;
  context: RecommendationContext;
  recommendation: {
    headline: string;
    summary: string;
    primaryZoneId: string;
    cautionZoneIds: string[];
    avoidZoneIds: string[];
  };
  zones: RecommendationZone[];
};

const EXPORT_LOG_PATH = "data/03_hasdeo_1000_all_bands/metadata/export_log.csv";

const FULL_BOUNDS: Bounds = [82.58, 22.4, 83.2, 22.9];
const CORE_BOUNDS: Bounds = [82.75, 22.5, 83.05, 22.8];

const ZONES: ZoneDefinition[] = [
  { id: "hasdeo-buffer", label: "Hasdeo Buffer", bounds: [82.5, 22.3, 83.3, 23.0], type: "buffer" },
  { id: "hasdeo-west", label: "Hasdeo West", bounds: [82.58, 22.4, 82.9, 22.9], type: "edge" },
  { id: "hasdeo-east", label: "Hasdeo East", bounds: [82.95, 22.4, 83.2, 22.9], type: "edge" },
  { id: "hasdeo-north", label: "Hasdeo North", bounds: [82.58, 22.65, 83.2, 22.9], type: "half" },
  { id: "hasdeo-south", label: "Hasdeo South", bounds: [82.58, 22.4, 83.2, 22.65], type: "half" },
  { id: "hasdeo-core", label: "Hasdeo Core", bounds: CORE_BOUNDS, type: "core" }
];

const METHODOLOGY = [
  {
    label: "Core overlap",
    weight: 0.4,
    description: "Zones overlapping the dense forest core are treated as environmentally sensitive."
  },
  {
    label: "Centrality",
    weight: 0.25,
    description: "Zones nearer the center of the Hasdeo block are penalized because they fragment interior forest."
  },
  {
    label: "Periphery advantage",
    weight: 0.2,
    description: "Outer-edge zones are favored because they avoid cutting through the center of the block."
  },
  {
    label: "Evidence confidence",
    weight: 0.15,
    description: "Monthly export coverage is used only as a confidence signal for the recommendation, not as ecological risk."
  }
] as const;

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value));
}

function mean(values: number[]) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function boundsArea(bounds: Bounds) {
  const [minLon, minLat, maxLon, maxLat] = bounds;
  const meanLat = (minLat + maxLat) / 2;
  const widthKm = (maxLon - minLon) * 111.32 * Math.cos((meanLat * Math.PI) / 180);
  const heightKm = (maxLat - minLat) * 110.57;
  return Math.max(0, widthKm * heightKm);
}

function centroid(bounds: Bounds) {
  return {
    lon: (bounds[0] + bounds[2]) / 2,
    lat: (bounds[1] + bounds[3]) / 2
  };
}

function intersectionArea(boundsA: Bounds, boundsB: Bounds) {
  const minLon = Math.max(boundsA[0], boundsB[0]);
  const minLat = Math.max(boundsA[1], boundsB[1]);
  const maxLon = Math.min(boundsA[2], boundsB[2]);
  const maxLat = Math.min(boundsA[3], boundsB[3]);
  if (minLon >= maxLon || minLat >= maxLat) return 0;
  return boundsArea([minLon, minLat, maxLon, maxLat]);
}

function distanceKm(a: { lon: number; lat: number }, b: { lon: number; lat: number }) {
  const lonKm = (a.lon - b.lon) * 111.32 * Math.cos((((a.lat + b.lat) / 2) * Math.PI) / 180);
  const latKm = (a.lat - b.lat) * 110.57;
  return Math.sqrt(lonKm * lonKm + latKm * latKm);
}

function zoneRegionName(zoneId: string) {
  return zoneId
    .replace("hasdeo-", "Hasdeo_")
    .replace(/(^|_)([a-z])/g, (_, prefix: string, letter: string) => `${prefix}${letter.toUpperCase()}`);
}

async function getEvidenceScores(): Promise<Map<string, number>> {
  // Read export_log.csv from R2 instead of local filesystem
  const rows = await fetchCsv(EXPORT_LOG_PATH);
  const byRegion = new Map<string, number[]>();

  for (const row of rows) {
    const phase = row.phase as PhaseKey;
    if (!["first_event", "second_event", "post_event"].includes(phase)) continue;
    const region = row.region;
    const sceneCount = Number(row.scene_count);
    if (!Number.isFinite(sceneCount)) continue;
    if (!byRegion.has(region)) byRegion.set(region, []);
    byRegion.get(region)?.push(sceneCount);
  }

  const averages = new Map<string, number>();
  const values: number[] = [];
  for (const [region, counts] of byRegion.entries()) {
    const average = mean(counts);
    averages.set(region, average);
    values.push(average);
  }

  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  return new Map(
    Array.from(averages.entries()).map(([region, value]) => [
      region,
      maxValue === minValue ? 100 : 60 + ((value - minValue) / (maxValue - minValue)) * 40
    ])
  );
}

async function buildContext(): Promise<RecommendationContext> {
  const features = await getFeatures();
  const pre = features.filter((row) => row.period >= "2016-10" && row.period <= "2017-12");
  const disturbance = features.filter((row) => row.period >= "2022-03" && row.period <= "2022-04");
  const post = features.filter((row) => row.period >= "2024-01" && row.period <= "2024-12");

  const ndviPre = mean(pre.map((row) => row.NDVI ?? 0));
  const ndviDisturbance = mean(disturbance.map((row) => row.NDVI ?? 0));
  const bsiPre = mean(pre.map((row) => row.BSI ?? 0));
  const bsiDisturbance = mean(disturbance.map((row) => row.BSI ?? 0));
  const heatPre = mean(pre.map((row) => row.LST_celsius ?? 0));
  const heatDisturbance = mean(disturbance.map((row) => row.LST_celsius ?? 0));
  const ndviPost = mean(post.map((row) => row.NDVI ?? 0));

  const ndviDropPct = ndviPre ? ((ndviDisturbance - ndviPre) / Math.abs(ndviPre)) * 100 : 0;
  const bsiRisePct = bsiPre ? ((bsiDisturbance - bsiPre) / Math.abs(bsiPre || 1)) * 100 : 0;
  const heatRisePct = heatPre ? ((heatDisturbance - heatPre) / Math.abs(heatPre)) * 100 : 0;

  return {
    ndviDropPct,
    bsiRisePct,
    heatRisePct,
    disturbanceWindow: "March-April 2022 shows the strongest exposed-land and thermal stress in the feature timeline.",
    recoverySignal:
      ndviPost >= ndviDisturbance
        ? "Later periods recover some vegetation signal, but the disturbance window still marks the critical planning caution."
        : "The later timeline does not fully recover the vegetation signal, so conservative siting is warranted."
  };
}

export async function getHasdeoRecommendationData(): Promise<RecommendationData> {
  const evidenceScores = await getEvidenceScores();
  const fullCenter = centroid(FULL_BOUNDS);
  const coreCenter = centroid(CORE_BOUNDS);
  const maxCenterDistance = distanceKm(centroid([FULL_BOUNDS[0], FULL_BOUNDS[1], FULL_BOUNDS[0], FULL_BOUNDS[1]]), fullCenter);

  const zones = ZONES.map((zone) => {
    const zoneCenter = centroid(zone.bounds);
    const zoneArea = boundsArea(zone.bounds);
    const coreOverlapPct = (intersectionArea(zone.bounds, CORE_BOUNDS) / zoneArea) * 100;
    const centerDistance = distanceKm(zoneCenter, fullCenter);
    const centralityPct = clamp01(1 - centerDistance / maxCenterDistance) * 100;
    const distanceFromCorePct = clamp01(distanceKm(zoneCenter, coreCenter) / maxCenterDistance) * 100;

    const [minLon, minLat, maxLon, maxLat] = zone.bounds;
    const edgeDistanceLon = Math.min(zoneCenter.lon - FULL_BOUNDS[0], FULL_BOUNDS[2] - zoneCenter.lon);
    const edgeDistanceLat = Math.min(zoneCenter.lat - FULL_BOUNDS[1], FULL_BOUNDS[3] - zoneCenter.lat);
    const maxEdgeDistanceLon = (FULL_BOUNDS[2] - FULL_BOUNDS[0]) / 2;
    const maxEdgeDistanceLat = (FULL_BOUNDS[3] - FULL_BOUNDS[1]) / 2;
    const interiorPenalty =
      ((edgeDistanceLon / maxEdgeDistanceLon) + (edgeDistanceLat / maxEdgeDistanceLat)) / 2;
    const peripheralAccessPct = clamp01(1 - interiorPenalty) * 100;

    const evidenceScore = evidenceScores.get(zoneRegionName(zone.id)) ?? 70;

    const suitabilityRaw =
      (100 - coreOverlapPct) * 0.4 +
      peripheralAccessPct * 0.25 +
      distanceFromCorePct * 0.2 +
      evidenceScore * 0.15;

    const typeAdjustment =
      zone.type === "buffer"
        ? 8
        : zone.type === "edge"
          ? 4
          : zone.type === "half"
            ? -3
            : -15;

    const suitabilityScore = Math.max(0, Math.min(100, suitabilityRaw + typeAdjustment));
    const environmentalRisk = 100 - suitabilityScore;

    const category =
      suitabilityScore >= 60
        ? "Preferred zone"
        : suitabilityScore >= 40
          ? "Conditional"
          : "Avoid";

    const facts = [
      `${coreOverlapPct.toFixed(1)}% overlap with the dense forest core.`,
      `${peripheralAccessPct.toFixed(1)}% periphery advantage relative to the full Hasdeo block.`,
      `${evidenceScore.toFixed(0)}/100 evidence confidence from monthly export coverage.`
    ];

    const rationale =
      category === "Preferred zone"
        ? `${zone.label} stays comparatively peripheral and keeps most development pressure away from the core forest.`
        : category === "Conditional"
          ? `${zone.label} may be considered only with stricter safeguards because it still carries meaningful ecological sensitivity.`
          : `${zone.label} overlaps too much of the ecological interior to be a lower-impact development choice.`;

    return {
      id: zone.id,
      label: zone.label,
      bounds: zone.bounds,
      centroid: zoneCenter,
      areaKm2: zoneArea,
      category,
      suitabilityScore,
      environmentalRisk,
      evidenceScore,
      coreOverlapPct,
      centralityPct,
      peripheralAccessPct,
      distanceFromCorePct,
      rationale,
      facts
    } satisfies RecommendationZone;
  }).sort((a, b) => b.suitabilityScore - a.suitabilityScore);

  const context = await buildContext();
  const primary = zones[0];
  const caution = zones
    .filter((zone) => zone.category === "Conditional" && zone.id !== primary.id)
    .map((zone) => zone.id);
  const avoid = zones.filter((zone) => zone.category === "Avoid").map((zone) => zone.id);

  return {
    generatedAt: new Date().toISOString(),
    studyArea: {
      label: "Hasdeo Arand planning zones",
      bounds: FULL_BOUNDS
    },
    methodology: METHODOLOGY.map((item) => ({ ...item })),
    context,
    recommendation: {
      headline: `${primary.label} emerges as the lower-impact candidate search area.`,
      summary:
        "This ranking favors peripheral zones with lower overlap into the dense forest core while keeping the evidence trail visible.",
      primaryZoneId: primary.id,
      cautionZoneIds: caution,
      avoidZoneIds: avoid
    },
    zones
  };
}