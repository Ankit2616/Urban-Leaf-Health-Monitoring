"""
03_augmentation.py
Urban Tree Monitoring System — Phase 3: Data Augmentation
Reads processed patches (.npy) → applies augmentations → saves expanded dataset.
Target: 4000-5000 augmented samples from ~1000 raw patches.
"""

import numpy as np
import random
import logging
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

# ─── CONFIG ────────────────────────────────────────────────────────────────────

PROCESSED_DIR  = Path("/Data/username/urban_tree_project/processed")
AUGMENTED_DIR  = Path("/Data/username/urban_tree_project/augmented")
LOG_DIR        = Path("/Data/username/urban_tree_project/logs")

AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

AUG_PER_PATCH  = 4    # 4 augmented versions per raw patch → ~4000-5000 total
PATCH_SIZE     = 256
N_CHANNELS     = 14   # Must match preprocessing output

# ─── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "03_augmentation.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── CUSTOM SPECTRAL AUGMENTATION ─────────────────────────────────────────────

class SpectralJitter(ImageOnlyTransform):
    """
    Random per-band multiplicative noise (simulates atmospheric variation).
    Applies only to spectral bands (indices 0-9), not vegetation indices.
    """
    def __init__(self, factor=0.05, n_spectral_bands=10, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.factor = factor
        self.n_bands = n_spectral_bands

    def apply(self, img, **params):
        img = img.copy()
        for i in range(self.n_bands):
            noise = 1.0 + np.random.uniform(-self.factor, self.factor)
            img[..., i] = img[..., i] * noise
        return img

    def get_transform_init_args_name(self):
        return ("factor", "n_spectral_bands")


class RandomBandDrop(ImageOnlyTransform):
    """
    Randomly zero out 1-2 non-critical bands (simulates sensor dropout).
    Never drops NDVI (index 10), EVI (11), SAVI (12), BSI (13).
    """
    def __init__(self, max_drop=2, always_apply=False, p=0.3):
        super().__init__(always_apply, p)
        self.max_drop = max_drop
        self.droppable = list(range(0, 10))  # Only optical bands

    def apply(self, img, **params):
        img = img.copy()
        n_drop = random.randint(1, self.max_drop)
        drop_bands = random.sample(self.droppable, min(n_drop, len(self.droppable)))
        for b in drop_bands:
            img[..., b] = 0.0
        return img

    def get_transform_init_args_name(self):
        return ("max_drop",)

# ─── AUGMENTATION PIPELINE ─────────────────────────────────────────────────────

def build_augmentation_pipeline():
    """
    Returns albumentations pipeline for multi-channel satellite patches.
    All geometric transforms are consistent across all bands.
    """
    return A.Compose([
        # --- Spatial Augmentations ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.75),
        A.Transpose(p=0.3),

        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=45,
            border_mode=0,     # zero padding
            p=0.5
        ),

        A.RandomCrop(height=PATCH_SIZE, width=PATCH_SIZE, p=1.0),

        # --- Pixel-level / Spectral ---
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # --- Radiometric Simulation ---
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5
        ),

        # --- Custom spectral augmentation ---
        SpectralJitter(factor=0.05, n_spectral_bands=10, p=0.4),
        RandomBandDrop(max_drop=1, p=0.2),

    ])

# ─── AUGMENT ONE PATCH ─────────────────────────────────────────────────────────

def augment_patch(patch, pipeline, n=AUG_PER_PATCH):
    """
    Apply n augmentation variants to a single patch.
    patch: (H, W, C) float32
    Returns: list of n augmented patches
    """
    results = []

    # Always keep original
    results.append(patch)

    for _ in range(n - 1):
        # albumentations needs float32 in [0,1] or arbitrary for custom transforms
        # We pass the multi-channel array directly
        augmented = pipeline(image=patch)['image']
        results.append(augmented)

    return results

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Urban Tree Monitoring — Data Augmentation")
    log.info("=" * 60)

    patch_files = sorted(PROCESSED_DIR.glob("patch_*.npy"))
    log.info(f"Found {len(patch_files)} processed patches")

    if len(patch_files) == 0:
        log.error("No patches found! Run 02_preprocessing.py first.")
        return

    pipeline = build_augmentation_pipeline()
    total_saved = 0

    for patch_path in tqdm(patch_files, desc="Augmenting patches"):
        try:
            patch = np.load(patch_path).astype(np.float32)

            # Validate shape
            if patch.shape != (PATCH_SIZE, PATCH_SIZE, N_CHANNELS):
                log.warning(f"Unexpected shape {patch.shape} in {patch_path.name}, skipping.")
                continue

            variants = augment_patch(patch, pipeline, n=AUG_PER_PATCH)

            for j, variant in enumerate(variants):
                stem   = patch_path.stem.replace("patch_", "")
                outpath = AUGMENTED_DIR / f"aug_{stem}_{j:02d}.npy"
                np.save(outpath, variant.astype(np.float32))
                total_saved += 1

        except Exception as e:
            log.error(f"Failed on {patch_path.name}: {e}")

    log.info("=" * 60)
    log.info(f"DONE: {total_saved} augmented patches saved to {AUGMENTED_DIR}")
    size_gb = total_saved * PATCH_SIZE * PATCH_SIZE * N_CHANNELS * 4 / 1e9
    log.info(f"Dataset size: ~{size_gb:.2f} GB")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
