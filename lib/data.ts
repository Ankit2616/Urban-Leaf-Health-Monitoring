import {
  getManifest,
  fetchCsv,
  r2Url,
} from "@/lib/r2";
import {
  FEATURE_KEYS,
  type FeatureRecord,
  type ImageRecord,
  type RegionSummary,
} from "@/lib/shared";
export { FEATURE_KEYS } from "@/lib/shared";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function toNumber(value: string): number | undefined {
  if (!value) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseDateFromName(filename: string): { date: string; year: number; month?: number } | null {
  const iso = filename.match(/(20\d{2}|19\d{2})[-_](\d{2})(?:[-_](\d{2}))?/);
  if (!iso) return null;
  const year = Number(iso[1]);
  const month = Number(iso[2]);
  const day = iso[3] ? Number(iso[3]) : 1;
  return {
    date: `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`,
    year,
    month,
  };
}

function inferRegion(filename: string): string {
  if (filename.includes("Control_Forest")) return "control-forest";
  if (filename.includes("Kente_Extension")) return "kente-extension";
  if (filename.includes("PEKB_Core")) return "pekb-core";
  if (filename.includes("Hasdeo_North")) return "hasdeo-north";
  if (filename.includes("Hasdeo_Full") || filename.includes("Hasdeo_Arand"))
    return "hasdeo-full";
  if (
    filename.includes("Blue_Mts") ||
    filename.includes("Sydney") ||
    filename.includes("Penrith") ||
    filename.includes("Warragamba")
  )
    return "sydney-blue-mountains";
  if (filename.includes("KI_")) return "kangaroo-island";
  return "hasdeo-full";
}

function inferPhase(filename: string, year: number): string {
  if (filename.includes("PREFIRE") || year <= 2017) return "baseline";
  if (filename.includes("DURING") || filename.includes("rapid_exp") || (year >= 2018 && year <= 2019))
    return "disturbance";
  if (filename.includes("slowdown") || year === 2020 || year === 2021 || year === 2022)
    return "active-monitoring";
  if (filename.includes("POST") || filename.includes("renewed") || year >= 2023)
    return "recovery";
  return "monitoring";
}

// ---------------------------------------------------------------------------
// Features — reads the main CSV from R2
// ---------------------------------------------------------------------------

export async function getFeatures(): Promise<FeatureRecord[]> {
  const rows = await fetchCsv(
    "data/02_comparison_based_on_events/event_1/csv_files/cleaned_images_features.csv"
  );
  return rows
    .map((row) => {
      const base: FeatureRecord = {
        id: row.id,
        lat: toNumber(row.lat) ?? 0,
        lon: toNumber(row.lon) ?? 0,
        period: row.period,
      };
      for (const key of FEATURE_KEYS) {
        const value = toNumber(row[key]);
        if (value !== undefined) base[key] = value;
      }
      return base;
    })
    .filter((row) => row.period);
}

// ---------------------------------------------------------------------------
// Images — built from the R2 manifest (no more filesystem walks)
// ---------------------------------------------------------------------------

function imagesFromPaths(
  paths: string[],
  source: "event_jpeg" | "sample_rgb" | "asset"
): ImageRecord[] {
  return paths
    .flatMap((relativePath): ImageRecord[] => {
      const filename = relativePath.split("/").pop() ?? relativePath;
      const parsed = parseDateFromName(filename);
      if (!parsed) return [];
      const region = inferRegion(filename);
      const image: ImageRecord = {
        id: Buffer.from(relativePath).toString("base64url"),
        region,
        date: parsed.date,
        year: parsed.year,
        phase: inferPhase(filename, parsed.year),
        source,
        filename,
        path: relativePath,
      };
      if (parsed.month !== undefined) image.month = parsed.month;
      return [image];
    })
    .sort((a, b) => a.date.localeCompare(b.date));
}

export async function getImages(): Promise<ImageRecord[]> {
  const manifest = await getManifest();
  const results: ImageRecord[] = [
    ...imagesFromPaths(manifest.event_jpeg, "event_jpeg"),
    ...imagesFromPaths(manifest.sample_rgb, "sample_rgb"),
    ...imagesFromPaths(manifest.asset, "asset"),
  ];
  return results.sort((a, b) => a.date.localeCompare(b.date));
}

// ---------------------------------------------------------------------------
// Regions
// ---------------------------------------------------------------------------

export async function getRegions(): Promise<RegionSummary[]> {
  const [images, features] = await Promise.all([getImages(), getFeatures()]);
  const ids = Array.from(new Set(images.map((image) => image.region)));
  if (!ids.includes("hasdeo-full")) ids.unshift("hasdeo-full");

  return ids.map((id) => {
    const regionImages = images.filter((image) => image.region === id);
    const dates = regionImages.map((image) => image.date).sort();
    return {
      id,
      label: id
        .split("-")
        .map((part) => part[0].toUpperCase() + part.slice(1))
        .join(" "),
      imageCount: regionImages.length,
      featureCount: id.startsWith("hasdeo") ? features.length : 0,
      start: dates[0],
      end: dates[dates.length - 1],
    };
  });
}

// ---------------------------------------------------------------------------
// Feature filtering & image lookup
// ---------------------------------------------------------------------------

export function filterFeatures(
  rows: FeatureRecord[],
  from: string,
  to: string
): FeatureRecord[] {
  return rows.filter((row) => row.period >= from && row.period <= to);
}

export function pickNearestImage(
  images: ImageRecord[],
  region: string,
  period: string
): ImageRecord | undefined {
  const target = new Date(
    `${period.length === 4 ? `${period}-01` : period}-01`
  ).getTime();
  const candidates = images.filter(
    (image) =>
      image.region === region ||
      (region === "hasdeo-full" && image.region.startsWith("hasdeo"))
  );
  return candidates
    .map((image) => ({
      image,
      distance: Math.abs(new Date(image.date).getTime() - target),
    }))
    .sort((a, b) => a.distance - b.distance)[0]?.image;
}

// ---------------------------------------------------------------------------
// Asset path — direct R2 URL (replaces local fs path resolution)
// ---------------------------------------------------------------------------

export function safeAssetPath(relativePath: string): string | null {
  return r2Url(relativePath);
}