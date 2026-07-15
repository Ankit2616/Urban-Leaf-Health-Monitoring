import { NextRequest, NextResponse } from "next/server";
import { filterFeatures, getFeatures } from "@/lib/data";
import { FEATURE_KEYS } from "@/lib/shared";

export async function GET(request: NextRequest) {
  const search = request.nextUrl.searchParams;
  const from = search.get("from") ?? "2016-10";
  const to = search.get("to") ?? "2025-12";
  const feature = search.get("feature");
  const rows = filterFeatures(await getFeatures(), from, to);

  return NextResponse.json({
    features: FEATURE_KEYS,
    selectedFeature: feature && FEATURE_KEYS.includes(feature as never) ? feature : "NDVI",
    rows
  });
}
