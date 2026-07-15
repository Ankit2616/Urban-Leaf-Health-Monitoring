#!/usr/bin/env node
/**
 * Upload local data and assets to Cloudflare R2 using the S3-compatible API.
 * Usage: node scripts/upload_to_r2.js
 */

const {
  S3Client,
  PutObjectCommand,
  ListObjectsV2Command,
} = require("@aws-sdk/client-s3");
const fs = require("fs");
const path = require("path");

// Read env vars from .env.local manually
const envPath = path.join(__dirname, "..", ".env.local");
const envContent = fs.readFileSync(envPath, "utf8");
const env = {};
envContent.split("\n").forEach((line) => {
  const trimmed = line.trim();
  if (!trimmed || trimmed.startsWith("#")) return;
  const idx = trimmed.indexOf("=");
  if (idx > -1) {
    env[trimmed.slice(0, idx).trim()] = trimmed.slice(idx + 1).trim();
  }
});

const BUCKET = "my-satellite-data-urban-leaf-monitoring";

const s3 = new S3Client({
  region: "auto",
  endpoint: env.R2_ENDPOINT,
  credentials: {
    accessKeyId: env.R2_ACCESS_KEY_ID,
    secretAccessKey: env.R2_SECRET_ACCESS_KEY,
  },
});

async function listExistingKeys() {
  const keys = new Set();
  let continuationToken;
  do {
    const res = await s3.send(
      new ListObjectsV2Command({
        Bucket: BUCKET,
        MaxKeys: 1000,
        ContinuationToken: continuationToken,
      })
    );
    for (const obj of res.Contents || []) {
      keys.add(obj.Key);
    }
    continuationToken = res.NextContinuationToken;
  } while (continuationToken);
  return keys;
}

async function uploadFile(filePath, key) {
  const body = fs.readFileSync(filePath);
  const ext = path.extname(filePath).toLowerCase();
  const contentTypes = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".csv": "text/csv",
    ".json": "application/json",
  };
  await s3.send(
    new PutObjectCommand({
      Bucket: BUCKET,
      Key: key,
      Body: body,
      ContentType: contentTypes[ext] || "application/octet-stream",
    })
  );
}

async function main() {
  const manifest = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "r2-manifest.json"), "utf8")
  );

  // Collect all paths from manifest
  const allPaths = [
    ...manifest.event_jpeg,
    ...manifest.sample_rgb,
    ...manifest.asset,
    ...manifest.csvFiles,
  ];

  // Also include r2-manifest.json itself
  allPaths.push("r2-manifest.json");

  console.log(`Total files to upload: ${allPaths.length}`);
  console.log("Checking existing files in R2...");

  let existingKeys;
  try {
    existingKeys = await listExistingKeys();
    console.log(`Already in R2: ${existingKeys.size} files`);
  } catch (e) {
    console.log(
      "Could not list existing files, will upload all:",
      e.message
    );
    existingKeys = new Set();
  }

  let uploaded = 0;
  let skipped = 0;
  let failed = 0;

  for (const relativePath of allPaths) {
    const localPath = path.join(__dirname, "..", relativePath);
    const r2Key = relativePath.replace(/^\/+/, "");

    if (existingKeys.has(r2Key)) {
      skipped++;
      continue;
    }

    if (!fs.existsSync(localPath)) {
      console.log(`  MISSING: ${relativePath}`);
      failed++;
      continue;
    }

    try {
      await uploadFile(localPath, r2Key);
      uploaded++;
      if (uploaded % 10 === 0) {
        console.log(`  Uploaded ${uploaded} files...`);
      }
    } catch (e) {
      console.log(`  FAILED: ${relativePath} - ${e.message}`);
      failed++;
    }
  }

  console.log(
    `\nDone! Uploaded: ${uploaded}, Skipped: ${skipped}, Failed: ${failed}`
  );
}

main().catch((e) => {
  console.error("Fatal error:", e);
  process.exit(1);
});