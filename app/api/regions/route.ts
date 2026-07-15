import { NextResponse } from "next/server";
import { getRegions } from "@/lib/data";

export async function GET() {
  return NextResponse.json({ regions: await getRegions() });
}
