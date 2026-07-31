/**
 * Cloudflare R2 client — replaces all local filesystem reads with HTTP fetches.
 *
 * Public files  → served directly through the R2 public bucket URL.
 * Private write  → uses the S3-compatible API (available via @aws-sdk/client-s3).
 */

let _publicUrl: string | null = null;
let _manifest: R2Manifest | null = null;

export type R2Manifest = {
  event_jpeg: string[];
  sample_rgb: string[];
  asset: string[];
  models?: {
    masks: string[];
    overlays: string[];
    summaries: string[];
  };
  csvFiles: string[];
};

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

function getPublicUrl(): string {
  if (!_publicUrl) {
    _publicUrl =
      process.env.R2_PUBLIC_URL ?? process.env.NEXT_PUBLIC_R2_PUBLIC_URL ?? "";
    if (!_publicUrl) {
      throw new Error(
        "Missing R2_PUBLIC_URL env var. Set it in .env.local or Vercel dashboard."
      );
    }
    _publicUrl = _publicUrl.replace(/\/+$/, ""); // strip trailing slash
  }
  return _publicUrl;
}

/** Accept a path relative to the R2 bucket root, return a fetchable URL. */
export function r2Url(relativePath: string): string {
  const base = getPublicUrl();
  const clean = relativePath.replace(/^\/+/, "");
  return `${base}/${clean}`;
}

// ---------------------------------------------------------------------------
// Manifest — describes every image / CSV file stored in R2
// ---------------------------------------------------------------------------

async function fetchManifest(): Promise<R2Manifest> {
  // Always fetch from R2 — avoids ENOENT on Vercel serverless
  const res = await fetch(r2Url("r2-manifest.json"));
  if (!res.ok)
    throw new Error(`Failed to fetch r2-manifest.json (HTTP ${res.status})`);
  return res.json();
}

export async function getManifest(): Promise<R2Manifest> {
  if (!_manifest) {
    _manifest = await fetchManifest();
  }
  return _manifest;
}

// ---------------------------------------------------------------------------
// CSV helpers — fetch a small CSV file from R2 and return parsed rows
// ---------------------------------------------------------------------------

function parseCsv(text: string): Record<string, string>[] {
  const rows: string[][] = [];
  let field = "";
  let row: string[] = [];
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"' && quoted && next === '"') {
      field += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(field);
      field = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(field);
      field = "";
      if (row.some((value) => value.trim() !== "")) rows.push(row);
      row = [];
    } else {
      field += char;
    }
  }

  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }

  const [headers, ...body] = rows;
  if (!headers) return [];
  return body.map((values) =>
    Object.fromEntries(
      headers.map((header, index) => [header.trim(), values[index]?.trim() ?? ""])
    )
  );
}

export async function fetchCsv(relativePath: string): Promise<Record<string, string>[]> {
  const url = r2Url(relativePath);
  const res = await fetch(url);
  if (!res.ok)
    throw new Error(`Failed to fetch ${relativePath} (HTTP ${res.status})`);
  const text = await res.text();
  return parseCsv(text);
}

// ---------------------------------------------------------------------------
// Model result helpers — fetch files from R2
// ---------------------------------------------------------------------------

export async function fetchJsonFromR2(relativePath: string): Promise<unknown> {
  const url = r2Url(relativePath);
  const res = await fetch(url);
  if (!res.ok) return null;
  return res.json();
}

export async function fetchBinaryFromR2(
  relativePath: string
): Promise<ArrayBuffer | null> {
  const url = r2Url(relativePath);
  const res = await fetch(url);
  if (!res.ok) return null;
  return res.arrayBuffer();
}

// ---------------------------------------------------------------------------
// S3 client for write operations (future use)
// ---------------------------------------------------------------------------

let _s3Client: import("@aws-sdk/client-s3").S3Client | null = null;

export function getS3Client(): import("@aws-sdk/client-s3").S3Client {
  if (!_s3Client) {
    const { S3Client } = require("@aws-sdk/client-s3") as typeof import("@aws-sdk/client-s3");
    _s3Client = new S3Client({
      region: "auto",
      endpoint: process.env.R2_ENDPOINT ?? "",
      credentials: {
        accessKeyId: process.env.R2_ACCESS_KEY_ID ?? "",
        secretAccessKey: process.env.R2_SECRET_ACCESS_KEY ?? "",
      },
    });
  }
  return _s3Client;
}