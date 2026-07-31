export type FeatureKey =
  | "NDVI"
  | "NDWI"
  | "MNDWI"
  | "NDBI"
  | "EVI"
  | "SAVI"
  | "LAI"
  | "GCI"
  | "ARVI"
  | "BSI"
  | "LST_celsius"
  | "canopy_cover_pct";

export type FeatureRecord = {
  id: string;
  lat: number;
  lon: number;
  period: string;
} & Partial<Record<FeatureKey, number>>;

export type ImageRecord = {
  id: string;
  region: string;
  date: string;
  year: number;
  month?: number;
  phase: string;
  source: "event_jpeg" | "sample_rgb" | "asset";
  filename: string;
  path: string;
};

export type RegionSummary = {
  id: string;
  label: string;
  imageCount: number;
  featureCount: number;
  start?: string;
  end?: string;
};

export const FEATURE_KEYS: FeatureKey[] = [
  "NDVI",
  "NDWI",
  "MNDWI",
  "NDBI",
  "EVI",
  "SAVI",
  "LAI",
  "GCI",
  "ARVI",
  "BSI",
  "LST_celsius",
  "canopy_cover_pct"
];

export const FEATURE_LABELS: Record<FeatureKey, string> = {
  NDVI: "Vegetation density",
  NDWI: "Water content",
  MNDWI: "Modified water index",
  NDBI: "Built-up intensity",
  EVI: "Enhanced vegetation",
  SAVI: "Soil-adjusted vegetation",
  LAI: "Leaf area index",
  GCI: "Green chlorophyll",
  ARVI: "Atmosphere-resistant vegetation",
  BSI: "Bare soil exposure",
  LST_celsius: "Land surface temperature",
  canopy_cover_pct: "Canopy cover"
};
