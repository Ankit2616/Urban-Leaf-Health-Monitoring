import { NextRequest, NextResponse } from "next/server";
import { getImages } from "@/lib/data";

export async function GET(request: NextRequest) {
  const search = request.nextUrl.searchParams;
  const region = search.get("region") ?? "hasdeo-full";
  const from = search.get("from") ?? "2016-01";
  const to = search.get("to") ?? "2025-12";
  const images = (await getImages()).filter((image) => {
    const period = image.date.slice(0, 7);
    const regionMatch = image.region === region || (region === "hasdeo-full" && image.region.startsWith("hasdeo"));
    return regionMatch && period >= from && period <= to;
  });

  return NextResponse.json({ images });
}
