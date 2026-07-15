"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";
import {
  Activity,
  ArrowDownRight,
  ArrowLeftRight,
  ArrowUpRight,
  CalendarRange,
  Cpu,
  Database,
  FileCheck2,
  GitBranch,
  Image as ImageIcon,
  Layers3,
  LineChart,
  MapPinned,
  RefreshCcw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  SquareStack,
  Thermometer,
  Trees,
  Waves
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { FeatureKey, FeatureRecord, ImageRecord, RegionSummary } from "@/lib/shared";
import { FEATURE_LABELS } from "@/lib/shared";

type CompareResponse = {
  region: string;
  from: string;
  to: string;
  feature: FeatureKey;
  before?: FeatureRecord;
  after?: FeatureRecord;
  delta: number | null;
  beforeImage?: ImageRecord;
  afterImage?: ImageRecord;
  timeline: FeatureRecord[];
};

type Props = {
  regions: RegionSummary[];
  featureKeys: FeatureKey[];
};

const FEATURE_GROUPS: Array<{ title: string; icon: LucideIcon; features: FeatureKey[] }> = [
  { title: "Vegetation", icon: Trees, features: ["NDVI", "EVI", "SAVI", "LAI", "GCI", "ARVI", "canopy_cover_pct"] },
  { title: "Water", icon: Waves, features: ["NDWI", "MNDWI"] },
  { title: "Urban and soil", icon: SquareStack, features: ["NDBI", "BSI"] },
  { title: "Thermal", icon: Thermometer, features: ["LST_celsius"] }
];

const FEATURE_COLORS: Record<FeatureKey, string> = {
  NDVI: "#1f7a4c",
  NDWI: "#27739b",
  MNDWI: "#48a2b8",
  NDBI: "#9b5c2e",
  EVI: "#54a36f",
  SAVI: "#7fb069",
  LAI: "#2f8f62",
  GCI: "#8fcf5f",
  ARVI: "#3f9f77",
  BSI: "#d39132",
  LST_celsius: "#d96a4e",
  canopy_cover_pct: "#6a9f58"
};

type ModelResults = {
  hasModel: boolean;
  hasInference: boolean;
  modelPath?: string;
  trainingHistory?: { epoch: number; val_iou?: number; val_acc?: number; train_loss?: number; val_loss?: number }[];
  summaries: Array<{ name: string; path: string; data: Record<string, { name: string; pixels: number; ratio: number }> }>;
  masks: Array<{ name: string; path: string }>;
  overlays: Array<{ name: string; path: string }>;
  expectedArtifacts?: string[];
  runbook?: {
    script: string;
    purpose: string;
    example: string;
  };
};

type SignalCard = {
  key: string;
  label: string;
  value: string;
  tone: "good" | "watch" | "risk" | "neutral";
  detail: string;
};

// ── Interactive Before/After Slider ──
function ComparisonSlider({ before, after }: { before?: ImageRecord; after?: ImageRecord }) {
  const [sliderPos, setSliderPos] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleMove = useCallback((clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  }, []);

  const onMouseDown = useCallback(() => setIsDragging(true), []);
  const onMouseUp = useCallback(() => setIsDragging(false), []);

  useEffect(() => {
    if (!isDragging) return;
    const onUp = () => setIsDragging(false);
    const onMove = (e: MouseEvent | TouchEvent) => {
      const clientX = "touches" in e ? e.touches[0].clientX : (e as MouseEvent).clientX;
      handleMove(clientX);
    };
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchend", onUp);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("touchmove", onMove);
    return () => {
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchend", onUp);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("touchmove", onMove);
    };
  }, [isDragging, handleMove]);

  if (!before && !after) {
    return <div className="status-line">No matching images for this timeline.</div>;
  }

  return (
    <div className="comparison-slider-container" ref={containerRef}>
      {/* BEFORE layer (full width) */}
      <div className="comparison-layer">
        {before ? (
          <Image
            src={`/api/asset?path=${encodeURIComponent(before.path)}`}
            alt={`Before: ${before.filename}`}
            fill
            unoptimized
            className="comparison-img"
          />
        ) : (
          <div className="comparison-placeholder">No before image</div>
        )}
        <div className="comparison-label before-label">{before?.date ?? "Before"}</div>
        {before && (
          <div className="comparison-meta">
            {before.region.replace("-", " ")} &middot; {before.phase.replace("-", " ")}
          </div>
        )}
      </div>

      {/* AFTER layer (clipped) */}
      <div
        className="comparison-layer comparison-after-clip"
        style={{ clipPath: `inset(0 0 0 ${sliderPos}%)` }}
        onMouseDown={onMouseDown}
        onTouchStart={onMouseDown}
      >
        {after ? (
          <Image
            src={`/api/asset?path=${encodeURIComponent(after.path)}`}
            alt={`After: ${after.filename}`}
            fill
            unoptimized
            className="comparison-img"
          />
        ) : (
          <div className="comparison-placeholder">No after image</div>
        )}
        <div className="comparison-label after-label">{after?.date ?? "After"}</div>
        {after && (
          <div className="comparison-meta">
            {after.region.replace("-", " ")} &middot; {after.phase.replace("-", " ")}
          </div>
        )}
      </div>

      {/* Slider handle */}
      <div className="comparison-handle" style={{ left: `${sliderPos}%` }} onMouseDown={onMouseDown} onTouchStart={onMouseDown}>
        <div className="comparison-handle-line" />
        <div className="comparison-handle-knob">
          <ArrowLeftRight size={16} />
        </div>
      </div>
    </div>
  );
}

// ── Helpers ──
function assetUrl(image?: ImageRecord) {
  return image ? `/api/asset?path=${encodeURIComponent(image.path)}` : "";
}

function formatValue(value?: number, digits = 3) {
  return value === undefined || Number.isNaN(value) ? "n/a" : value.toFixed(digits);
}

function featureTone(feature: FeatureKey, delta: number | null) {
  if (delta === null) return "No comparable values available.";
  const direction = delta >= 0 ? "increased" : "decreased";
  if (feature === "NDVI" || feature === "EVI" || feature === "SAVI" || feature === "GCI") {
    return delta >= 0
      ? `Vegetation signal ${direction}; this points to recovery or denser canopy.`
      : `Vegetation signal ${direction}; this supports degradation or canopy loss.`;
  }
  if (feature === "BSI" || feature === "NDBI" || feature === "LST_celsius") {
    return delta >= 0
      ? `${FEATURE_LABELS[feature]} ${direction}; this can indicate exposed land, heat, or built-up pressure.`
      : `${FEATURE_LABELS[feature]} ${direction}; this can indicate reduced disturbance pressure.`;
  }
  return `${FEATURE_LABELS[feature]} ${direction} across the selected window.`;
}

function metricDelta(before?: FeatureRecord, after?: FeatureRecord, key?: FeatureKey) {
  if (!before || !after || !key) return null;
  const beforeValue = before[key];
  const afterValue = after[key];
  return typeof beforeValue === "number" && typeof afterValue === "number" ? afterValue - beforeValue : null;
}

function formatDelta(value: number | null, digits = 2) {
  if (value === null) return "n/a";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function buildSignalCards(data: CompareResponse | null): SignalCard[] {
  const ndvi = metricDelta(data?.before, data?.after, "NDVI");
  const canopy = metricDelta(data?.before, data?.after, "canopy_cover_pct");
  const heat = metricDelta(data?.before, data?.after, "LST_celsius");
  const urban = metricDelta(data?.before, data?.after, "NDBI");
  const soil = metricDelta(data?.before, data?.after, "BSI");
  const water = metricDelta(data?.before, data?.after, "NDWI");

  return [
    {
      key: "vegetation",
      label: "Vegetation health",
      value: formatDelta(ndvi, 3),
      tone: ndvi === null ? "neutral" : ndvi >= 0 ? "good" : ndvi > -0.04 ? "watch" : "risk",
      detail: ndvi === null ? "No NDVI comparison available." : "NDVI change across the selected timeline."
    },
    {
      key: "canopy",
      label: "Canopy condition",
      value: formatDelta(canopy, 2),
      tone: canopy === null ? "neutral" : canopy >= 0 ? "good" : canopy > -5 ? "watch" : "risk",
      detail: canopy === null ? "No canopy estimate available." : "Estimated canopy cover shift from the feature table."
    },
    {
      key: "heat",
      label: "Heat pressure",
      value: formatDelta(heat, 2),
      tone: heat === null ? "neutral" : heat <= 0 ? "good" : heat < 2 ? "watch" : "risk",
      detail: heat === null ? "No LST comparison available." : "Land surface temperature delta in Celsius."
    },
    {
      key: "urban",
      label: "Built-up pressure",
      value: formatDelta(urban, 3),
      tone: urban === null ? "neutral" : urban <= 0 ? "good" : urban < 0.04 ? "watch" : "risk",
      detail: urban === null ? "No NDBI comparison available." : "NDBI captures built-up intensity in the selected period."
    },
    {
      key: "soil",
      label: "Bare-soil exposure",
      value: formatDelta(soil, 3),
      tone: soil === null ? "neutral" : soil <= 0 ? "good" : soil < 0.05 ? "watch" : "risk",
      detail: soil === null ? "No BSI comparison available." : "Bare Soil Index highlights exposed or disturbed ground."
    },
    {
      key: "water",
      label: "Moisture signal",
      value: formatDelta(water, 3),
      tone: water === null ? "neutral" : water >= 0 ? "good" : water > -0.04 ? "watch" : "risk",
      detail: water === null ? "No NDWI comparison available." : "NDWI change helps explain vegetation stress."
    }
  ];
}

function assessmentText(cards: SignalCard[]) {
  const riskCount = cards.filter((card) => card.tone === "risk").length;
  const watchCount = cards.filter((card) => card.tone === "watch").length;
  if (riskCount >= 2) return "High-change window: multiple signals indicate canopy stress, heat, bare soil, or built-up pressure.";
  if (riskCount || watchCount >= 2) return "Watch window: the selected timeline shows early warning signals that deserve inspection.";
  return "Stable or improving window: the selected indicators do not show broad degradation pressure.";
}

function Chart({ rows, feature }: { rows: FeatureRecord[]; feature: FeatureKey }) {
  const points = useMemo(() => {
    const values = rows
      .map((row) => ({ period: row.period, value: row[feature] }))
      .filter((row): row is { period: string; value: number } => typeof row.value === "number");
    const min = Math.min(...values.map((row) => row.value));
    const max = Math.max(...values.map((row) => row.value));
    const range = max - min || 1;
    return values.map((row, index) => ({
      ...row,
      x: values.length === 1 ? 40 : 42 + (index / (values.length - 1)) * 716,
      y: 250 - ((row.value - min) / range) * 190,
      min,
      max
    }));
  }, [feature, rows]);

  if (!points.length) {
    return <div className="status-line">No values available for the selected feature and range.</div>;
  }

  const pathData = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  const first = points[0];
  const last = points[points.length - 1];

  return (
    <svg className="chart" viewBox="0 0 800 300" role="img" aria-label={`${feature} timeline chart`}>
      <line x1="42" y1="250" x2="758" y2="250" stroke="#cdd6ca" />
      <line x1="42" y1="58" x2="42" y2="250" stroke="#cdd6ca" />
      {[0, 1, 2, 3].map((tick) => (
        <line key={tick} x1="42" y1={250 - tick * 48} x2="758" y2={250 - tick * 48} stroke="#edf1ea" />
      ))}
      <path d={pathData} fill="none" stroke="#25633f" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
      {points.map((point) => (
        <circle key={point.period} cx={point.x} cy={point.y} r="4" fill="#b77226" />
      ))}
      <text x="42" y="38">{formatValue(first.max)}</text>
      <text x="42" y="274">{first.period}</text>
      <text x="704" y="274">{last.period}</text>
      <text x="710" y="38">{formatValue(first.min)}</text>
    </svg>
  );
}

function MiniFeatureChart({ rows, feature }: { rows: FeatureRecord[]; feature: FeatureKey }) {
  const points = useMemo(() => {
    const values = rows
      .map((row) => ({ period: row.period, value: row[feature] }))
      .filter((row): row is { period: string; value: number } => typeof row.value === "number");
    if (!values.length) return [];
    const min = Math.min(...values.map((row) => row.value));
    const max = Math.max(...values.map((row) => row.value));
    const range = max - min || 1;
    return values.map((row, index) => ({
      ...row,
      x: values.length === 1 ? 6 : 6 + (index / (values.length - 1)) * 188,
      y: 72 - ((row.value - min) / range) * 54,
      min,
      max
    }));
  }, [feature, rows]);

  if (!points.length) return <div className="mini-empty">No data</div>;
  const pathData = points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  const first = points[0];
  const last = points[points.length - 1];
  const delta = last.value - first.value;

  return (
    <article className="mini-feature-card">
      <div className="mini-feature-head">
        <div>
          <strong>{feature}</strong>
          <span>{FEATURE_LABELS[feature]}</span>
        </div>
        <em className={delta < 0 ? "negative" : "positive"}>{delta >= 0 ? "+" : ""}{delta.toFixed(2)}</em>
      </div>
      <svg viewBox="0 0 200 84" aria-label={`${feature} mini chart`}>
        <path d="M 6 72 H 194" stroke="#e0e6e0" />
        <path d={pathData} fill="none" stroke={FEATURE_COLORS[feature]} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx={points[0].x} cy={points[0].y} r="3" fill={FEATURE_COLORS[feature]} />
        <circle cx={last.x} cy={last.y} r="3" fill={FEATURE_COLORS[feature]} />
      </svg>
    </article>
  );
}

function AnalysisSummary({ cards, region, from, to }: { cards: SignalCard[]; region?: string; from: string; to: string }) {
  return (
    <section className="surface intelligence-surface">
      <div className="surface-head">
        <div>
          <h2>Monitoring intelligence</h2>
          <p className="muted">
            Parameter-driven assessment for {region ?? "selected region"} from {from} to {to}.
          </p>
        </div>
        <span className="badge insight-badge">
          <ShieldCheck size={14} aria-hidden />
          No upload required
        </span>
      </div>
      <div className="assessment-strip">
        <LineChart size={20} aria-hidden />
        <strong>{assessmentText(cards)}</strong>
      </div>
      <div className="signal-grid">
        {cards.map((card) => (
          <article className={`signal-card ${card.tone}`} key={card.key}>
            <span>{card.label}</span>
            <strong>{card.value}</strong>
            <p>{card.detail}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function PipelinePanel({ modelResults }: { modelResults: ModelResults | null }) {
  const steps = [
    {
      title: "Curated satellite archive",
      body: "The user selects a region and timeline from prepared monthly scenes and feature tables.",
      icon: Database
    },
    {
      title: "Feature comparison API",
      body: "The backend returns vegetation, canopy, water, soil, built-up, and thermal signals for that window.",
      icon: Layers3
    },
    {
      title: "Model artifact bridge",
      body: modelResults?.hasInference
        ? "Inference artifacts are available and can be shown beside the feature timelines."
        : "The dashboard is waiting for checkpoint inference masks, overlays, and class summaries.",
      icon: Cpu
    },
    {
      title: "Decision dashboard",
      body: "The UI presents before/after imagery, trend charts, risk signals, and model outputs in one workflow.",
      icon: FileCheck2
    }
  ];

  return (
    <section className="surface pipeline-surface">
      <div className="surface-head">
        <div>
          <h2>How this system works</h2>
          <p className="muted">This is a timeline-monitoring product, not a random image upload demo.</p>
        </div>
      </div>
      <div className="pipeline-grid">
        {steps.map((step, index) => {
          const Icon = step.icon;
          return (
            <article className="pipeline-step" key={step.title}>
              <div>
                <Icon size={18} aria-hidden />
                <span>{String(index + 1).padStart(2, "0")}</span>
              </div>
              <strong>{step.title}</strong>
              <p>{step.body}</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function ModelResultsPanel({ modelResults }: { modelResults: ModelResults | null }) {
  const latestSummary = modelResults?.summaries[0];
  const latestEpoch = modelResults?.trainingHistory?.at(-1);
  return (
    <section className="surface model-surface">
      <div className="surface-head">
        <div>
          <h2>Model output lane</h2>
          <p className="muted">Segmentation masks, overlays, and class ratios appear here after curated-scene inference.</p>
        </div>
        <span className={modelResults?.hasInference ? "badge success-badge" : "badge warning-badge"}>
          {modelResults?.hasInference ? "Inference ready" : "Awaiting inference"}
        </span>
      </div>

      <div className="model-status-grid">
        <div>
          <span>Checkpoint</span>
          <strong>{modelResults?.hasModel ? "Detected" : "Missing"}</strong>
          <small>{modelResults?.modelPath ?? "Expected under results/ or h100_config/results/"}</small>
        </div>
        <div>
          <span>Inference artifacts</span>
          <strong>{modelResults?.hasInference ? "Detected" : "Missing"}</strong>
          <small>{modelResults ? `${modelResults.masks.length} masks, ${modelResults.overlays.length} overlays` : "No artifact scan available"}</small>
        </div>
        <div>
          <span>Latest validation</span>
          <strong>
            {latestEpoch?.val_iou !== undefined
              ? `${(latestEpoch.val_iou * 100).toFixed(1)}% IoU`
              : latestEpoch?.val_acc !== undefined
                ? `${(latestEpoch.val_acc * 100).toFixed(1)}% acc`
                : "n/a"}
          </strong>
          <small>{latestEpoch ? `epoch ${latestEpoch.epoch}` : "Training history not found"}</small>
        </div>
      </div>

      {modelResults?.hasInference && latestSummary ? (
        <div className="model-grid">
          <div className="model-preview">
            {modelResults.overlays[0] || modelResults.masks[0] ? (
              <Image
                src={`/api/asset?path=${encodeURIComponent((modelResults.overlays[0] ?? modelResults.masks[0]).path)}`}
                alt="Model segmentation output"
                width={900}
                height={560}
                unoptimized
              />
            ) : null}
          </div>
          <div className="model-classes">
            {Object.entries(latestSummary.data).map(([key, value]) => (
              <div key={key} className="class-row">
                <span>{value.name}</span>
                <strong>{(value.ratio * 100).toFixed(1)}%</strong>
                <div>
                  <i style={{ width: `${Math.max(2, value.ratio * 100)}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="model-empty">
          <GitBranch size={28} aria-hidden />
          <div>
            <strong>No model inference results found in this repo yet.</strong>
<p>
  Currently displaying curated GEE/CSV feature data. To visualize your trained model in real time, run <code>{modelResults?.runbook?.script ?? "h100_config/06_predict_visualize.py"}</code>   on your selected scenes. The API will automatically sync your masks, overlays, and JSON summaries.
</p>
            {modelResults?.runbook?.example ? (
              /* FORCE TEXT BREAKING HERE */
              <code className="run-command !block !w-full !whitespace-pre-wrap !break-all !normal-case tracking-normal overflow-hidden">
                {modelResults.runbook.example}
              </code>
            ) : null}
          </div>
        </div>
      )}
    </section>
  );
}

export default function TimelineDashboard({ regions, featureKeys }: Props) {
  const defaultRegion = regions.find((item) => item.id === "hasdeo-full")?.id ?? regions[0]?.id ?? "hasdeo-full";
  const [region, setRegion] = useState(defaultRegion);
  const [feature, setFeature] = useState<FeatureKey>("NDVI");
  const [from, setFrom] = useState("2018-01");
  const [to, setTo] = useState("2024-12");
  const [data, setData] = useState<CompareResponse | null>(null);
  const [modelResults, setModelResults] = useState<ModelResults | null>(null);
  const [loading, setLoading] = useState(true);

  async function loadComparison(overrides?: Partial<{ region: string; feature: FeatureKey; from: string; to: string }>) {
    setLoading(true);
    const nextRegion = overrides?.region ?? region;
    const nextFeature = overrides?.feature ?? feature;
    const nextFrom = overrides?.from ?? from;
    const nextTo = overrides?.to ?? to;
    const params = new URLSearchParams({ region: nextRegion, feature: nextFeature, from: nextFrom, to: nextTo });
    const response = await fetch(`/api/compare?${params.toString()}`);
    setData(await response.json());
    setLoading(false);
  }

  useEffect(() => {
    void loadComparison();
    fetch("/api/model-results")
      .then((response) => response.json())
      .then(setModelResults)
      .catch(() => setModelResults(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const beforeValue = data?.before?.[feature];
  const afterValue = data?.after?.[feature];
  const delta = data?.delta ?? null;
  const selectedRegion = regions.find((item) => item.id === region);
  const deltaDirection = delta === null ? "neutral" : delta < 0 ? "down" : "up";
  const changePct =
    beforeValue !== undefined && afterValue !== undefined && beforeValue !== 0
      ? ((afterValue - beforeValue) / Math.abs(beforeValue)) * 100
      : null;
  const signalCards = buildSignalCards(data);

  return (
    <main className="page">
      <section className="workspace-head">
        <div className="hero-copy">
          <span className="eyebrow premium-eyebrow">Remote sensing intelligence</span>
          <h1>Urban leaf health, mapped across time.</h1>
          <p>
            A premium monitoring workspace for comparing curated satellite imagery, vegetation indices, and
            disturbance signals across user-selected timelines.
          </p>
          <div className="hero-actions">
            <button className="btn hero-btn" onClick={() => void loadComparison()}>
              <Search size={17} aria-hidden />
              Analyze Timeline
            </button>
            <span className="hero-proof">
              <Sparkles size={15} aria-hidden />
              Hasdeo, Sydney fringe, Kangaroo Island
            </span>
          </div>
        </div>
        <div className="hero-visual" aria-label="Satellite monitoring preview">
          <div className="hero-image hero-image-main" />
          <div className="hero-image hero-image-small one" />
          <div className="hero-image hero-image-small two" />
          <div className="hero-glass">
            <span>Current signal</span>
            <strong>{feature}</strong>
            <small>{featureTone(feature, delta)}</small>
          </div>
        </div>
        <div className="head-metrics" aria-label="Dataset summary">
          <div className="metric-tile">
            <Database size={18} aria-hidden />
            <span>Feature rows</span>
            <strong>{selectedRegion?.featureCount ?? 0}</strong>
          </div>
          <div className="metric-tile">
            <ImageIcon size={18} aria-hidden />
            <span>Curated imagery</span>
            <strong>{selectedRegion?.imageCount ?? 0}</strong>
          </div>
          <div className="metric-tile">
            <MapPinned size={18} aria-hidden />
            <span>Active region</span>
            <strong>{selectedRegion?.label ?? "Hasdeo Full"}</strong>
          </div>
        </div>
      </section>

      <div className="tool-grid">
        <aside className="sidebar">
          <div className="sidebar-title">
            <SlidersHorizontal size={18} aria-hidden />
            <div>
              <h2>Query controls</h2>
              <p>Region, feature, and month range</p>
            </div>
          </div>

          <div className="control">
            <label htmlFor="region">Region</label>
            <select id="region" value={region} onChange={(event) => setRegion(event.target.value)}>
              {regions.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label} ({item.imageCount})
                </option>
              ))}
            </select>
          </div>

          <div className="control">
            <label htmlFor="feature">Feature</label>
            <select id="feature" value={feature} onChange={(event) => setFeature(event.target.value as FeatureKey)}>
              {featureKeys.map((key) => (
                <option key={key} value={key}>
                  {key} - {FEATURE_LABELS[key]}
                </option>
              ))}
            </select>
          </div>

          <div className="control">
            <label htmlFor="from">From month</label>
            <input id="from" type="month" value={from} onChange={(event) => setFrom(event.target.value)} />
          </div>

          <div className="control">
            <label htmlFor="to">To month</label>
            <input id="to" type="month" value={to} onChange={(event) => setTo(event.target.value)} />
          </div>

          <div className="button-row">
            <button className="btn primary" onClick={() => void loadComparison()}>
              <Search size={17} aria-hidden />
              Compare
            </button>
            <button
              className="btn"
              onClick={() => {
                const preset = { region: "hasdeo-full", feature: "NDVI" as FeatureKey, from: "2022-01", to: "2022-04" };
                setRegion(preset.region);
                setFeature(preset.feature);
                setFrom(preset.from);
                setTo(preset.to);
                void loadComparison(preset);
              }}
            >
              <CalendarRange size={17} aria-hidden />
              Event
            </button>
          </div>

          <div className="status-line">
            <RefreshCcw size={15} aria-hidden />
            {loading ? "Loading project data..." : `${data?.timeline.length ?? 0} feature records in range`}
          </div>

          <div className="preset-panel">
            <span className="eyebrow">Fast windows</span>
            <button
              type="button"
              onClick={() => {
                const preset = { region: "hasdeo-full", feature: "BSI" as FeatureKey, from: "2022-01", to: "2022-04" };
                setRegion(preset.region);
                setFeature(preset.feature);
                setFrom(preset.from);
                setTo(preset.to);
                void loadComparison(preset);
              }}
            >
              Hasdeo exposed soil
            </button>
            <button
              type="button"
              onClick={() => {
                const preset = { region: "hasdeo-full", feature: "LST_celsius" as FeatureKey, from: "2022-01", to: "2022-04" };
                setRegion(preset.region);
                setFeature(preset.feature);
                setFrom(preset.from);
                setTo(preset.to);
                void loadComparison(preset);
              }}
            >
              Heat stress check
            </button>
          </div>
        </aside>

        <section className="content">
          <div className="metrics">
            <div className="stat accent-green">
              <span><Activity size={15} aria-hidden /> Before</span>
              <strong>{formatValue(beforeValue)}</strong>
              <small>{data?.before?.period ?? from}</small>
            </div>
            <div className="stat accent-blue">
              <span><Activity size={15} aria-hidden /> After</span>
              <strong>{formatValue(afterValue)}</strong>
              <small>{data?.after?.period ?? to}</small>
            </div>
            <div className={`stat accent-${deltaDirection}`}>
              <span>{delta !== null && delta < 0 ? <ArrowDownRight size={15} aria-hidden /> : <ArrowUpRight size={15} aria-hidden />} Delta</span>
              <strong>
                {delta === null ? "n/a" : `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`}
              </strong>
              <small>{changePct === null ? FEATURE_LABELS[feature] : `${changePct >= 0 ? "+" : ""}${changePct.toFixed(1)}% vs start`}</small>
            </div>
            <div className="stat accent-amber">
              <span><MapPinned size={15} aria-hidden /> Signal</span>
              <strong>{feature}</strong>
              <small>{featureTone(feature, delta)}</small>
            </div>
          </div>

          <AnalysisSummary cards={signalCards} region={selectedRegion?.label} from={from} to={to} />

          <section className="surface chart-surface">
            <div className="surface-head">
              <div>
                <h2>{feature} feature timeline</h2>
                <p className="muted">{FEATURE_LABELS[feature]} from {from} to {to}</p>
              </div>
              <div className="badge-row">
                <span className="badge">{selectedRegion?.label ?? region}</span>
                <span className="badge muted-badge">{data?.timeline.length ?? 0} points</span>
              </div>
            </div>
            <Chart rows={data?.timeline ?? []} feature={feature} />
          </section>

          <section className="surface">
            <div className="surface-head">
              <div>
                <h2>All factor signals</h2>
                <p className="muted">Vegetation, canopy, water, soil, urban, and heat features from the same API response.</p>
              </div>
            </div>
            <div className="factor-groups">
              {FEATURE_GROUPS.map((group) => {
                const Icon = group.icon;
                return (
                  <div className="factor-group" key={group.title}>
                    <div className="factor-group-head">
                      <Icon size={17} aria-hidden />
                      <strong>{group.title}</strong>
                    </div>
                    <div className="mini-feature-grid">
                      {group.features.map((item) => (
                        <MiniFeatureChart key={item} rows={data?.timeline ?? []} feature={item} />
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <PipelinePanel modelResults={modelResults} />

          <ModelResultsPanel modelResults={modelResults} />

          {/* ── Image Comparison with Interactive Slider ── */}
          <section className="surface">
            <div className="surface-head">
              <div>
                <h2>Image comparison</h2>
                <p className="muted">
                  Drag the slider left and right to compare the nearest satellite imagery for the selected start and end points.
                  <strong> Region/phase labels</strong> are shown on each side.
                </p>
              </div>
            </div>
            <ComparisonSlider before={data?.beforeImage} after={data?.afterImage} />
            {data?.beforeImage && data?.afterImage && (
              <div className="comparison-info-row">
                <span>
                  <strong>Before:</strong> {data.beforeImage.date} &mdash; Phase: {data.beforeImage.phase.replace("-", " ")}
                </span>
                <span>
                  <strong>After:</strong> {data.afterImage.date} &mdash; Phase: {data.afterImage.phase.replace("-", " ")}
                </span>
                <span>
                  <strong>Filename:</strong> {data.beforeImage.filename} &harr; {data.afterImage.filename}
                </span>
              </div>
            )}
          </section>

          <section className="surface">
            <div className="surface-head">
              <div>
                <h2>Feature details</h2>
                <p className="muted">Monthly records returned by the feature API.</p>
              </div>
            </div>
            <div className="table-shell">
              <table>
                <thead>
                  <tr>
                    <th>Period</th>
                    <th>NDVI</th>
                    <th>EVI</th>
                    <th>SAVI</th>
                    <th>BSI</th>
                    <th>NDBI</th>
                    <th>LST</th>
                  </tr>
                </thead>
                <tbody>
                  {(data?.timeline ?? []).slice(-18).map((row) => (
                    <tr key={row.period}>
                      <td>{row.period}</td>
                      <td>{formatValue(row.NDVI)}</td>
                      <td>{formatValue(row.EVI)}</td>
                      <td>{formatValue(row.SAVI)}</td>
                      <td>{formatValue(row.BSI)}</td>
                      <td>{formatValue(row.NDBI)}</td>
                      <td>{formatValue(row.LST_celsius, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </section>
      </div>
    </main>
  );
}