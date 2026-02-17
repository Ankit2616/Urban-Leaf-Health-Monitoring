"""
02_preprocessing.py
Urban Tree Monitoring System — Phase 2: Preprocessing
Reads raw multi-band TIFFs → cloud QC → patch extraction → normalize → save as .npy
"""

import os
import numpy as np
import rasterio
import json
import logging
from pathlib import Path
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────────

RAW_DIR       = Path("/Data/username/urban_tree_project/raw_tiff")
PROCESSED_DIR = Path("/Data/username/urban_tree_project/processed")
LOG_DIR       = Path("/Data/username/urban_tree_project/logs")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE    = 256           # Patch size for model input (pixels)
STRIDE        = 128           # Overlap stride (50% overlap = better coverage)
MIN_VALID     = 0.85          # At least 85% valid (non-masked) pixels per patch
BANDS_USED    = 14            # B2..B12 (10) + NDVI + EVI + SAVI + BSI (4)

# Band order in TIFF (matches 01_data_collection.py)
BAND_NAMES = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
              'NDVI','EVI','SAVI','BSI']

# ─── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "02_preprocessing.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── NORMALIZATION ─────────────────────────────────────────────────────────────

# Per-band statistics (computed from a representative subset)
# Update these values after running compute_stats() on your actual dataset
BAND_STATS = {
    # band_name: (mean, std)
    'B2':   (0.0500, 0.0300), 'B3': (0.0800, 0.0350),
    'B4':   (0.0750, 0.0400), 'B5': (0.1100, 0.0500),
    'B6':   (0.2200, 0.0800), 'B7': (0.2500, 0.0900),
    'B8':   (0.2700, 0.1000), 'B8A':(0.2800, 0.1050),
    'B11':  (0.1500, 0.0700), 'B12':(0.0900, 0.0500),
    'NDVI': (0.4000, 0.2000), 'EVI': (0.3000, 0.1500),
    'SAVI': (0.3500, 0.1800), 'BSI': (-0.1000, 0.2000),
}

def normalize_patch(patch):
    """
    Z-score normalize each band independently.
    patch: (H, W, C) float32
    Returns: (H, W, C) float32
    """
    normed = np.zeros_like(patch, dtype=np.float32)
    for i, name in enumerate(BAND_NAMES):
        mu, sigma = BAND_STATS[name]
        normed[..., i] = (patch[..., i] - mu) / (sigma + 1e-8)
    return normed

def compute_stats(tiff_files, n_sample=50):
    """Compute per-band mean/std from a sample of TIFFs (run once to calibrate)."""
    log.info(f"Computing band statistics from {min(n_sample, len(tiff_files))} files...")
    all_pixels = {name: [] for name in BAND_NAMES}

    for tiff in tiff_files[:n_sample]:
        with rasterio.open(tiff) as src:
            data = src.read().astype(np.float32)   # (C, H, W)
            # Get nodata mask
            nodata = src.nodata or 0
            valid = data[0] != nodata
            for i, name in enumerate(BAND_NAMES):
                if i < data.shape[0]:
                    band_vals = data[i][valid].flatten()
                    # Sample to avoid memory issues
                    if len(band_vals) > 100000:
                        idx = np.random.choice(len(band_vals), 100000, replace=False)
                        band_vals = band_vals[idx]
                    all_pixels[name].extend(band_vals.tolist())

    stats = {}
    for name, vals in all_pixels.items():
        arr = np.array(vals)
        stats[name] = (float(arr.mean()), float(arr.std()))
        log.info(f"  {name}: mean={stats[name][0]:.4f}, std={stats[name][1]:.4f}")

    # Save stats for reproducibility
    stats_path = PROCESSED_DIR / "band_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    log.info(f"Stats saved to {stats_path}")
    return stats

# ─── PATCH EXTRACTION ──────────────────────────────────────────────────────────

def extract_patches(data, mask, patch_size=PATCH_SIZE, stride=STRIDE, min_valid=MIN_VALID):
    """
    Extract valid patches from a multi-band image.
    data:  (C, H, W) float32
    mask:  (H, W) bool — True = valid pixel
    Returns: list of (patch_size, patch_size, C) arrays
    """
    C, H, W = data.shape
    patches = []

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            mask_patch = mask[y:y+patch_size, x:x+patch_size]

            # Skip patches with too many masked/invalid pixels
            valid_ratio = mask_patch.sum() / (patch_size * patch_size)
            if valid_ratio < min_valid:
                continue

            patch = data[:, y:y+patch_size, x:x+patch_size]  # (C, H, W)
            patch = np.transpose(patch, (1, 2, 0))             # → (H, W, C)

            # Fill remaining masked pixels with band median
            for c in range(C):
                band = patch[..., c]
                valid_vals = band[mask_patch]
                if len(valid_vals) > 0:
                    median_val = np.median(valid_vals)
                    band[~mask_patch] = median_val
                patch[..., c] = band

            patches.append(patch.astype(np.float32))

    return patches

# ─── PROCESS ONE TIFF ──────────────────────────────────────────────────────────

def process_tiff(tiff_path):
    """Process a single TIFF: read → QC → patch → normalize → return patches."""
    try:
        with rasterio.open(tiff_path) as src:
            data  = src.read().astype(np.float32)   # (C, H, W)
            nodata = src.nodata if src.nodata is not None else 0

            # Build valid pixel mask
            # A pixel is valid if it's not nodata in the first band
            valid_mask = (data[0] != nodata) & np.isfinite(data[0])

            # Clip to physical range (reflectance 0-1 for optical bands, -1 to 1 for indices)
            for i, name in enumerate(BAND_NAMES):
                if 'NDVI' in name or 'EVI' in name or 'SAVI' in name or 'BSI' in name:
                    data[i] = np.clip(data[i], -1, 1)
                else:
                    data[i] = np.clip(data[i], 0, 1)

    except Exception as e:
        log.error(f"Failed to read {tiff_path.name}: {e}")
        return []

    patches = extract_patches(data, valid_mask)

    # Normalize each patch
    normed_patches = [normalize_patch(p) for p in patches]

    return normed_patches

# ─── SAVE PATCHES ──────────────────────────────────────────────────────────────

def save_patches(patches, output_dir, prefix, start_idx):
    saved = 0
    for i, patch in enumerate(patches):
        filename = output_dir / f"{prefix}_{start_idx + i:06d}.npy"
        np.save(filename, patch)
        saved += 1
    return saved

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Urban Tree Monitoring — Preprocessing Pipeline")
    log.info("=" * 60)

    tiff_files = sorted(RAW_DIR.glob("*.tif"))
    log.info(f"Found {len(tiff_files)} TIFF files in {RAW_DIR}")

    if len(tiff_files) == 0:
        log.error("No TIFF files found! Run 01_data_collection.py first.")
        return

    # Step 1: Compute normalization statistics
    global BAND_STATS
    stats_path = PROCESSED_DIR / "band_statistics.json"
    if stats_path.exists():
        log.info("Loading existing band statistics...")
        with open(stats_path) as f:
            loaded = json.load(f)
        BAND_STATS = {k: tuple(v) for k, v in loaded.items()}
    else:
        stats = compute_stats(tiff_files, n_sample=min(100, len(tiff_files)))
        BAND_STATS = stats

    # Step 2: Extract and save patches
    total_patches = 0

    for tiff_path in tqdm(tiff_files, desc="Processing TIFFs"):
        patches = process_tiff(tiff_path)
        if patches:
            n = save_patches(patches, PROCESSED_DIR, "patch", total_patches)
            total_patches += n

    log.info("=" * 60)
    log.info(f"DONE: {total_patches} patches saved to {PROCESSED_DIR}")
    log.info(f"Each patch: {PATCH_SIZE}×{PATCH_SIZE}×{BANDS_USED} float32")
    log.info(f"Dataset size: ~{total_patches * PATCH_SIZE * PATCH_SIZE * BANDS_USED * 4 / 1e9:.2f} GB")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
