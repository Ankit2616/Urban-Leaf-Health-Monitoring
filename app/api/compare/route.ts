import { NextRequest, NextResponse } from "next/server";
import {
  filterFeatures,
  getFeatures,
  getImages,
  pickNearestImage
} from "@/lib/data";
import { FEATURE_KEYS, type FeatureKey } from "@/lib/shared";

function nearestFeature(rows: Awaited<ReturnType<typeof getFeatures>>, period: string) {
  const target = new Date(`${period}-01`).getTime();
  return rows
    .map((row) => ({ row, distance: Math.abs(new Date(`${row.period}-01`).getTime() - target) }))
    .sort((a, b) => a.distance - b.distance)[0]?.row;
}

export async function GET(request: NextRequest) {
  const search = request.nextUrl.searchParams;
  const region = search.get("region") ?? "hasdeo-full";
  const from = search.get("from") ?? "2018-01";
  const to = search.get("to") ?? "2024-12";
  const feature = (search.get("feature") ?? "NDVI") as FeatureKey;
  const selected = FEATURE_KEYS.includes(feature) ? feature : "NDVI";

  const [features, images] = await Promise.all([getFeatures(), getImages()]);
  const timeline = filterFeatures(features, from, to);
  const before = nearestFeature(features, from);
  const after = nearestFeature(features, to);
  const beforeValue = before?.[selected];
  const afterValue = after?.[selected];
  const delta = beforeValue !== undefined && afterValue !== undefined ? afterValue - beforeValue : null;

  return NextResponse.json({
    region,
    from,
    to,
    feature: selected,
    before,
    after,
    delta,
    beforeImage: pickNearestImage(images, region, from),
    afterImage: pickNearestImage(images, region, to),
    timeline
  });
}
