import { NextResponse } from "next/server";
import { getManifest, fetchJsonFromR2, r2Url } from "@/lib/r2";

export const dynamic = "force-dynamic";

export async function GET() {
  const manifest = await getManifest();
  const modelFiles = manifest.models;

  // Check if we have model directories listed in the manifest
  const hasModel = false; // Model checkpoints (.pth/.pt/.ckpt) aren't tracked in manifest
  const masks = (modelFiles?.masks ?? []).map((relativePath) => ({
    name: relativePath.split("/").pop() ?? relativePath,
    path: relativePath,
    url: r2Url(relativePath),
  }));
  const overlays = (modelFiles?.overlays ?? []).map((relativePath) => ({
    name: relativePath.split("/").pop() ?? relativePath,
    path: relativePath,
    url: r2Url(relativePath),
  }));

  // Try to fetch summary JSONs and training history
  const summaries = [];
  for (const relativePath of modelFiles?.summaries ?? []) {
    const data = await fetchJsonFromR2(relativePath);
    if (data) {
      const parsed = data as Record<string, unknown>;
      const values = Object.values(parsed);
      const firstValue = values[0] as unknown;
      const summaryData =
        firstValue &&
        typeof firstValue === "object" &&
        "ratio" in firstValue &&
        "pixels" in firstValue
          ? parsed
          : firstValue && typeof firstValue === "object"
            ? firstValue
            : parsed;
      summaries.push({
        name: relativePath.split("/").pop() ?? relativePath,
        path: relativePath,
        data: summaryData,
      });
    }
  }

  // Training history — attempt to fetch from known R2 paths
  let trainingHistory = null;
  const historyPaths = [
    "results/training_history.json",
    "h100_config/results/training_history.json",
  ];
  for (const historyPath of historyPaths) {
    const data = await fetchJsonFromR2(historyPath);
    if (data) {
      trainingHistory = data;
      break;
    }
  }

  return NextResponse.json({
    hasModel,
    hasInference: summaries.length > 0 || masks.length > 0 || overlays.length > 0,
    modelPath: null,
    trainingHistory,
    summaries,
    masks,
    overlays,
    expectedArtifacts: [
      "trained checkpoint: results/best_model.pth or h100_config/results/best_model.pth",
      "inference masks: *_mask.png",
      "visual overlays: *_overlay.png",
      "class summaries: *summary.json",
    ],
    runbook: {
      script: "h100_config/06_predict_visualize.py",
      purpose:
        "Batch inference over curated satellite rasters, then publish masks, overlays, and class ratios to the dashboard.",
      example:
        "python h100_config/06_predict_visualize.py --checkpoint results/best_model.pth --input data/<scene>.npy --output results/inference",
    },
  });
}