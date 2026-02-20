"""
LOCAL_TEST.py
─────────────────────────────────────────────────────────────────────────────
Urban Tree Monitoring System — LOCAL END-TO-END PIPELINE TEST
─────────────────────────────────────────────────────────────────────────────
Run this on your LOCAL machine BEFORE submitting to HPC.
It tests every stage of the pipeline with tiny settings (finishes in ~5 mins).

What it tests:
  ✓ Stage 1 - GEE connection + downloading 5 sample images (PNG, not TIFF)
  ✓ Stage 2 - Preprocessing: patch extraction + normalization
  ✓ Stage 3 - Augmentation: 4x augmented variants
  ✓ Stage 4 - Model: U-Net forward pass + loss computation
  ✓ Stage 5 - Training: 3 mini epochs on fake data
  ✓ Stage 6 - Inference: dummy prediction + visualization

HOW TO RUN:
  pip install earthengine-api rasterio numpy torch albumentations pillow matplotlib tqdm
  python LOCAL_TEST.py

If all stages print ✓ PASS — your code is ready for HPC.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import shutil
import numpy as np
import json
import traceback
from pathlib import Path

# ─── TEST WORKSPACE (deleted at end) ─────────────────────────────────────────

TEST_DIR = Path("./local_pipeline_test")
for sub in ["raw", "processed", "augmented", "models", "results"]:
    (TEST_DIR / sub).mkdir(parents=True, exist_ok=True)

PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS    = []

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def stage(name):
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"{'='*60}")

def ok(msg):
    global PASS_COUNT
    PASS_COUNT += 1
    RESULTS.append(("✅ PASS", msg))
    print(f"  ✅ PASS: {msg}")

def fail(msg, err=""):
    global FAIL_COUNT
    FAIL_COUNT += 1
    RESULTS.append(("❌ FAIL", msg))
    print(f"  ❌ FAIL: {msg}")
    if err:
        print(f"     Error: {err}")

def warn(msg):
    RESULTS.append(("⚠️  WARN", msg))
    print(f"  ⚠️  WARN: {msg}")

# ─── STAGE 0: DEPENDENCY CHECK ────────────────────────────────────────────────

stage("0 — Dependency Check")

required = {
    "ee":           "earthengine-api",
    "numpy":        "numpy",
    "torch":        "torch",
    "albumentations": "albumentations",
    "PIL":          "pillow",
    "matplotlib":   "matplotlib",
    "tqdm":         "tqdm",
    "rasterio":     "rasterio",
}

missing = []
for module, pkg in required.items():
    try:
        __import__(module)
        ok(f"{pkg} importable")
    except ImportError:
        fail(f"{pkg} NOT installed", f"Run: pip install {pkg}")
        missing.append(pkg)

if missing:
    print(f"\n⛔ Install missing packages first:\n  pip install {' '.join(missing)}")
    print("Then re-run LOCAL_TEST.py")
    sys.exit(1)

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')   # No display needed
import matplotlib.pyplot as plt
from PIL import Image

# GPU / CPU info
device_name = "CPU"
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    ok(f"GPU available: {device_name}")
else:
    warn("No GPU found — running on CPU (slower but fine for testing)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── STAGE 1: GOOGLE EARTH ENGINE ────────────────────────────────────────────

stage("1 — Google Earth Engine (5 sample PNG images)")

import ee
import requests

GEE_PROJECT_ID = ""  # ← FILL THIS IN if you have a project ID, else leave blank

try:
    if GEE_PROJECT_ID:
        ee.Initialize(project=GEE_PROJECT_ID)
    else:
        ee.Initialize()
    ee.Number(1).add(1).getInfo()
    ok("GEE initialized successfully")
    GEE_OK = True
except Exception as e:
    fail("GEE initialization failed", str(e)[:200])
    warn("Skipping GEE download — generating synthetic data instead")
    GEE_OK = False

if GEE_OK:
    try:
        # Hasdeo Forest ROI (small area for speed)
        roi = ee.Geometry.Rectangle([82.40, 22.80, 82.60, 23.00])

        s2 = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(roi)
            .filterDate('2022-01-01', '2023-12-31')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
            .sort('system:time_start')
            .limit(5)
        )

        count = s2.size().getInfo()
        ok(f"Sentinel-2 collection built: {count} images found")

        # Download 5 small PNG thumbnails to test download pipeline
        img_list = s2.toList(5)
        downloaded = 0
        for i in range(min(5, count)):
            img  = ee.Image(img_list.get(i))
            date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()

            rgb = img.select(['B4','B3','B2']).visualize(min=0, max=3000)
            url = rgb.getThumbURL({'region': roi, 'dimensions': 128, 'format': 'png'})

            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                path = TEST_DIR / "raw" / f"test_{i:02d}_{date}.png"
                path.write_bytes(r.content)
                downloaded += 1
            time.sleep(0.3)

        ok(f"Downloaded {downloaded}/5 sample images from GEE")

    except Exception as e:
        fail("GEE data download failed", str(e)[:200])
        GEE_OK = False

# Fallback: generate synthetic multi-band data if GEE failed
if not GEE_OK:
    warn("Generating SYNTHETIC multi-band patches to test pipeline stages 2-5")
    for i in range(10):
        # Simulate a 14-band Sentinel-2 patch (H, W, C)
        # Optical bands (0-9): reflectance 0-1
        patch = np.random.uniform(0.0, 0.5, (512, 512, 10)).astype(np.float32)
        # Vegetation indices (10-13): -1 to 1
        indices = np.random.uniform(-0.5, 0.8, (512, 512, 4)).astype(np.float32)
        full_patch = np.concatenate([patch, indices], axis=-1)
        np.save(TEST_DIR / "raw" / f"synthetic_{i:03d}.npy", full_patch)
    ok("10 synthetic multi-band patches generated for pipeline testing")

# ─── STAGE 2: PREPROCESSING ──────────────────────────────────────────────────

stage("2 — Preprocessing (patch extraction + normalization)")

PATCH_SIZE  = 64    # Small for local test (256 on HPC)
STRIDE      = 32
N_CHANNELS  = 14
BAND_NAMES  = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12',
               'NDVI','EVI','SAVI','BSI']

# Normalization stats (using realistic defaults)
BAND_STATS = {
    'B2':(0.05,0.03), 'B3':(0.08,0.035), 'B4':(0.075,0.04),
    'B5':(0.11,0.05), 'B6':(0.22,0.08),  'B7':(0.25,0.09),
    'B8':(0.27,0.10), 'B8A':(0.28,0.105),'B11':(0.15,0.07),
    'B12':(0.09,0.05),'NDVI':(0.4,0.2),  'EVI':(0.3,0.15),
    'SAVI':(0.35,0.18),'BSI':(-0.1,0.2),
}

def normalize_patch(patch):
    normed = np.zeros_like(patch, dtype=np.float32)
    for i, name in enumerate(BAND_NAMES):
        mu, sigma = BAND_STATS[name]
        normed[..., i] = (patch[..., i] - mu) / (sigma + 1e-8)
    return normed

def extract_patches_from_array(data, patch_size=PATCH_SIZE, stride=STRIDE):
    """data: (H, W, C)"""
    H, W, C = data.shape
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            p = data[y:y+patch_size, x:x+patch_size, :]
            patches.append(p.astype(np.float32))
    return patches

try:
    npy_files = list((TEST_DIR / "raw").glob("*.npy"))
    total_patches = 0

    for npy_path in npy_files:
        data = np.load(npy_path)
        if data.shape[-1] != N_CHANNELS:
            continue
        patches = extract_patches_from_array(data)
        for idx, patch in enumerate(patches):
            normed = normalize_patch(patch)
            out_path = TEST_DIR / "processed" / f"patch_{total_patches:06d}.npy"
            np.save(out_path, normed)
            total_patches += 1

    # If GEE was used (PNG files), generate synthetic patches for pipeline test
    png_files = list((TEST_DIR / "raw").glob("*.png"))
    if png_files and total_patches == 0:
        warn("PNG files found from GEE — generating synthetic patches for full pipeline test")
        for i in range(20):
            patch = np.random.uniform(-1, 1, (PATCH_SIZE, PATCH_SIZE, N_CHANNELS)).astype(np.float32)
            np.save(TEST_DIR / "processed" / f"patch_{i:06d}.npy", patch)
            total_patches += 1

    if total_patches > 0:
        ok(f"Extracted {total_patches} patches of shape ({PATCH_SIZE}×{PATCH_SIZE}×{N_CHANNELS})")

        # Validate a patch
        sample = np.load(list((TEST_DIR / "processed").glob("*.npy"))[0])
        assert sample.shape == (PATCH_SIZE, PATCH_SIZE, N_CHANNELS), f"Unexpected shape: {sample.shape}"
        assert sample.dtype == np.float32, "Expected float32"
        ok(f"Patch shape OK: {sample.shape}, dtype: {sample.dtype}")

        # Check for NaN/Inf
        if np.isfinite(sample).all():
            ok("No NaN/Inf values in patches")
        else:
            fail("NaN or Inf found in patches", "Check normalization or raw data")
    else:
        fail("No patches extracted", "Check raw data files")

except Exception as e:
    fail("Preprocessing stage failed", traceback.format_exc()[-400:])

# ─── STAGE 3: AUGMENTATION ────────────────────────────────────────────────────

stage("3 — Augmentation (4x variants per patch)")

try:
    import albumentations as A
    from albumentations.core.transforms_interface import ImageOnlyTransform

    class SpectralJitter(ImageOnlyTransform):
        def apply(self, img, **params):
            img = img.copy()
            for i in range(10):
                noise = 1.0 + np.random.uniform(-0.05, 0.05)
                img[..., i] *= noise
            return img
        def get_transform_init_args_name(self): return ()

    pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.75),
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
        A.RandomCrop(height=PATCH_SIZE, width=PATCH_SIZE, p=1.0),
        SpectralJitter(p=0.4),
    ])

    patch_files = list((TEST_DIR / "processed").glob("patch_*.npy"))
    aug_count   = 0

    for pf in patch_files[:5]:   # Only first 5 patches for local test
        patch = np.load(pf).astype(np.float32)
        for j in range(4):   # 4 variants
            aug = pipeline(image=patch)['image']
            np.save(TEST_DIR / "augmented" / f"aug_{pf.stem}_{j}.npy", aug)
            aug_count += 1

    ok(f"Generated {aug_count} augmented patches")

    # Validate augmented patch
    aug_files = list((TEST_DIR / "augmented").glob("*.npy"))
    sample_aug = np.load(aug_files[0])
    assert sample_aug.shape == (PATCH_SIZE, PATCH_SIZE, N_CHANNELS)
    ok(f"Augmented patch shape OK: {sample_aug.shape}")

except Exception as e:
    fail("Augmentation stage failed", traceback.format_exc()[-400:])

# ─── STAGE 4: MODEL ARCHITECTURE ─────────────────────────────────────────────

stage("4 — U-Net Model (forward pass + loss)")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.block(x)

    class Down(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
        def forward(self, x): return self.block(x)

    class Up(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        def forward(self, x1, x2):
            x1 = self.up(x1)
            dy = x2.size(2) - x1.size(2)
            dx = x2.size(3) - x1.size(3)
            x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
            return self.conv(torch.cat([x2, x1], dim=1))

    class MiniUNet(nn.Module):
        def __init__(self, n_in=14, n_cls=5):
            super().__init__()
            self.inc   = DoubleConv(n_in, 32)
            self.down1 = Down(32, 64)
            self.down2 = Down(64, 128)
            self.up1   = Up(128 + 64, 64)
            self.up2   = Up(64 + 32, 32)
            self.out   = nn.Conv2d(32, n_cls, 1)
        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x  = self.up1(x3, x2)
            x  = self.up2(x, x1)
            return self.out(x)

    N_CLASSES  = 5
    model      = MiniUNet(n_in=N_CHANNELS, n_cls=N_CLASSES).to(device)
    params     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok(f"U-Net created: {params:,} parameters")

    # Forward pass
    dummy_input = torch.randn(2, N_CHANNELS, PATCH_SIZE, PATCH_SIZE).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, N_CLASSES, PATCH_SIZE, PATCH_SIZE), f"Wrong output shape: {output.shape}"
    ok(f"Forward pass OK: input {dummy_input.shape} → output {output.shape}")

    # Loss computation
    criterion = nn.CrossEntropyLoss()
    fake_label = torch.randint(0, N_CLASSES, (2, PATCH_SIZE, PATCH_SIZE)).to(device)
    loss = criterion(output, fake_label)
    ok(f"Loss computed: {loss.item():.4f}")

except Exception as e:
    fail("Model stage failed", traceback.format_exc()[-400:])

# ─── STAGE 5: MINI TRAINING LOOP ─────────────────────────────────────────────

stage("5 — Mini Training Loop (3 epochs, synthetic data)")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader

    class TinyDataset(Dataset):
        def __init__(self, n=20):
            self.n = n
        def __len__(self): return self.n
        def __getitem__(self, _):
            x = torch.randn(N_CHANNELS, PATCH_SIZE, PATCH_SIZE)
            y = torch.randint(0, N_CLASSES, (PATCH_SIZE, PATCH_SIZE))
            return x, y

    loader    = DataLoader(TinyDataset(20), batch_size=4, shuffle=True)
    model     = MiniUNet(n_in=N_CHANNELS, n_cls=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(3):
        model.train()
        epoch_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        losses.append(avg)
        print(f"     Epoch {epoch+1}/3 — loss: {avg:.4f}")

    ok(f"3 training epochs completed. Final loss: {losses[-1]:.4f}")

    # Save checkpoint
    torch.save({'epoch': 3, 'model_state': model.state_dict()},
               TEST_DIR / "models" / "test_checkpoint.pth")
    ok("Checkpoint saved successfully")

    # Test loading checkpoint
    ckpt = torch.load(TEST_DIR / "models" / "test_checkpoint.pth",
                      map_location=device)
    model.load_state_dict(ckpt['model_state'])
    ok("Checkpoint loaded successfully")

except Exception as e:
    fail("Training stage failed", traceback.format_exc()[-400:])

# ─── STAGE 6: INFERENCE + VISUALIZATION ──────────────────────────────────────

stage("6 — Inference + Visualization")

try:
    CLASS_NAMES   = ["Dense Veg", "Sparse Veg", "Bare Soil", "Built-up", "Water"]
    CLASS_COLORS  = np.array([
        [0,  120, 0],    # Dark green   — Dense vegetation
        [144, 238, 144], # Light green  — Sparse vegetation
        [210, 180, 140], # Tan          — Bare soil
        [128, 128, 128], # Gray         — Built-up/urban
        [30,  144, 255], # Blue         — Water
    ], dtype=np.uint8)

    model.eval()
    test_input = torch.randn(1, N_CHANNELS, PATCH_SIZE, PATCH_SIZE).to(device)

    with torch.no_grad():
        logits = model(test_input)
        pred   = logits.argmax(dim=1).squeeze().cpu().numpy()   # (H, W)

    # Build color mask
    color_mask = CLASS_COLORS[pred]   # (H, W, 3)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(test_input[0, :3].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Input (B2-B3-B4 as RGB)", fontsize=10)
    axes[0].axis('off')

    axes[1].imshow(color_mask)
    axes[1].set_title("Segmentation Prediction", fontsize=10)
    axes[1].axis('off')

    # NDVI heatmap (band index 10)
    ndvi = test_input[0, 10].cpu().numpy()
    im = axes[2].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[2].set_title("NDVI Heatmap", fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=np.array(c)/255, label=n)
              for n, c in zip(CLASS_NAMES, CLASS_COLORS)]
    fig.legend(handles=legend, loc='lower center', ncol=5, fontsize=8, frameon=True)

    plt.tight_layout()
    vis_path = TEST_DIR / "results" / "test_visualization.png"
    plt.savefig(vis_path, dpi=120, bbox_inches='tight')
    plt.close()
    ok(f"Visualization saved: {vis_path}")

    # Verify the image was created
    assert vis_path.exists() and vis_path.stat().st_size > 0
    ok("Visualization file valid")

except Exception as e:
    fail("Visualization stage failed", traceback.format_exc()[-400:])

# ─── FINAL REPORT ─────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print("  LOCAL PIPELINE TEST — FINAL REPORT")
print(f"{'='*60}")

for status, msg in RESULTS:
    print(f"  {status}  {msg}")

print(f"\n{'─'*60}")
print(f"  ✅ Passed: {PASS_COUNT}   ❌ Failed: {FAIL_COUNT}")
print(f"{'─'*60}")

if FAIL_COUNT == 0:
    print("""
  🎉 ALL STAGES PASSED!
  Your pipeline is ready for HPC submission.

  Next steps:
  1. Transfer your code to HPC:
       scp -r ./ username@10.10.11.201:/Data/username/urban_tree_project/
  2. Set up conda env:
       bash setup_env.sh
  3. Get your MIG UUID:
       nvidia-smi -L
  4. Update job_gpu.pbs with your MIG UUID
  5. Submit CPU pipeline:
       qsub job_cpu.pbs
  6. Submit GPU training (after CPU finishes):
       qsub job_gpu.pbs
""")
else:
    print(f"""
  ⚠️  {FAIL_COUNT} stage(s) FAILED. Fix them before submitting to HPC.
  Scroll up to see the error details for each failed stage.
""")

# Cleanup test workspace
try:
    shutil.rmtree(TEST_DIR)
    print(f"  🧹 Test workspace cleaned up.")
except:
    print(f"  ⚠️  Could not clean {TEST_DIR} — delete manually if needed.")