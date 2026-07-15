import { NextRequest, NextResponse } from "next/server";
import { safeAssetPath } from "@/lib/data";

const CONTENT_TYPES: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
};

export async function GET(request: NextRequest) {
  const relativePath = request.nextUrl.searchParams.get("path");
  if (!relativePath) {
    return NextResponse.json({ error: "Missing path" }, { status: 400 });
  }

  // safeAssetPath now returns a full R2 URL instead of a local filesystem path
  const assetUrl = safeAssetPath(relativePath);
  if (!assetUrl) {
    return NextResponse.json({ error: "Asset not allowed" }, { status: 403 });
  }

  // Fetch the asset from R2 and stream it back
  try {
    const res = await fetch(assetUrl);
    if (!res.ok) {
      return NextResponse.json(
        { error: "Asset not found on R2" },
        { status: 404 }
      );
    }

    const buffer = await res.arrayBuffer();
    const ext = relativePath
      .slice(relativePath.lastIndexOf("."))
      .toLowerCase();
    return new NextResponse(new Uint8Array(buffer), {
      headers: {
        "Content-Type": CONTENT_TYPES[ext] ?? "application/octet-stream",
        "Cache-Control": "public, max-age=3600",
      },
    });
  } catch {
    return NextResponse.json(
      { error: "Failed to fetch asset from R2" },
      { status: 502 }
    );
  }
}