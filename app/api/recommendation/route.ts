import { NextResponse } from "next/server";
import { getHasdeoRecommendationData } from "@/lib/recommendation";

export const dynamic = "force-dynamic";

export async function GET() {
  return NextResponse.json(await getHasdeoRecommendationData());
}
