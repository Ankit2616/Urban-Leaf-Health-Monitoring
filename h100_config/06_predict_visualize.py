"""
06_predict_visualize.py
Urban Tree Monitoring System — Phase 6: Inference and Visualization

Runs a trained Attention U-Net checkpoint on processed .npy patches or raw
multi-band GeoTIFF scenes, then writes class masks, color previews, and summary
metrics that can be used in reports or downstream GIS work.
"""

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast


N_CHANNELS = 14
N_CLASSES = 5
PATCH_SIZE = 256
STRIDE = 128
BAND_NAMES = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
    "NDVI", "EVI", "SAVI", "BSI",
]
CLASS_MAP = {
    0: "Vegetation",
    1: "Sparse Vegetation",
    2: "Bare Soil/Rock",
    3: "Built-up/Urban",
    4: "Water/Shadow",
}
CLASS_COLORS = np.array([
    [27, 128, 69],
    [166, 217, 106],
    [217, 172, 84],
    [120, 120, 120],
    [43, 131, 186],
], dtype=np.uint8)
DEFAULT_STATS = {
    "B2": (0.0500, 0.0300), "B3": (0.0800, 0.0350), "B4": (0.0750, 0.0400),
    "B5": (0.1100, 0.0500), "B6": (0.2200, 0.0800), "B7": (0.2500, 0.0900),
    "B8": (0.2700, 0.1000), "B8A": (0.2800, 0.1050), "B11": (0.1500, 0.0700),
    "B12": (0.0900, 0.0500), "NDVI": (0.4000, 0.2000), "EVI": (0.3000, 0.1500),
    "SAVI": (0.3500, 0.1800), "BSI": (-0.1000, 0.2000),
}


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_model_module():
    module_path = Path(__file__).with_name("04_model.py")
    spec = importlib.util.spec_from_file_location("urban_leaf_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Run vegetation segmentation inference.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pth.")
    parser.add_argument("--input", type=Path, required=True, help="A .npy/.tif file or a folder of files.")
    parser.add_argument("--output-dir", type=Path, default=Path("/Data/username/urban_tree_project/results/inference"))
    parser.add_argument("--stats", type=Path, default=None, help="band_statistics.json from preprocessing.")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def load_stats(path):
    if path and path.exists():
        with open(path) as f:
            loaded = json.load(f)
        return {k: tuple(v) for k, v in loaded.items()}
    return DEFAULT_STATS


def normalize_patch(patch, stats):
    normed = np.zeros_like(patch, dtype=np.float32)
    for idx, band in enumerate(BAND_NAMES):
        mean, std = stats[band]
        normed[..., idx] = (patch[..., idx] - mean) / (std + 1e-8)
    return normed


def colorize(mask):
    return CLASS_COLORS[np.clip(mask, 0, len(CLASS_COLORS) - 1)]


def stretch_rgb(rgb):
    out = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(3):
        arr = np.nan_to_num(rgb[..., channel].astype(np.float32))
        valid = arr[np.isfinite(arr)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, (2, 98))
        if hi <= lo:
            continue
        out[..., channel] = np.clip((arr - lo) / (hi - lo), 0, 1)
    return out


def summarize_mask(mask):
    counts = np.bincount(mask.reshape(-1), minlength=N_CLASSES)[:N_CLASSES]
    total = int(counts.sum())
    return {
        str(cls): {
            "name": CLASS_MAP[cls],
            "pixels": int(counts[cls]),
            "ratio": float(counts[cls] / total) if total else 0.0,
        }
        for cls in range(N_CLASSES)
    }


def load_checkpoint(checkpoint_path, device):
    model_module = load_model_module()
    model = model_module.UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_batch(model, patches, device, batch_size):
    predictions = []
    for start in range(0, len(patches), batch_size):
        batch = np.stack(patches[start:start + batch_size]).astype(np.float32)
        tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(device)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(tensor)
        pred = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        predictions.extend(pred)
    return predictions


def iter_windows(height, width, patch_size, stride):
    y_positions = list(range(0, max(height - patch_size + 1, 1), stride))
    x_positions = list(range(0, max(width - patch_size + 1, 1), stride))
    if not y_positions or y_positions[-1] != height - patch_size:
        y_positions.append(max(height - patch_size, 0))
    if not x_positions or x_positions[-1] != width - patch_size:
        x_positions.append(max(width - patch_size, 0))
    for y in y_positions:
        for x in x_positions:
            yield y, x


def predict_array(model, array, device, batch_size, patch_size, stride):
    height, width, channels = array.shape
    if channels != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} channels, found {channels}")

    padded_h = max(height, patch_size)
    padded_w = max(width, patch_size)
    padded = np.zeros((padded_h, padded_w, channels), dtype=np.float32)
    padded[:height, :width] = array

    votes = np.zeros((padded_h, padded_w, N_CLASSES), dtype=np.uint16)
    coords, patches = [], []
    for y, x in iter_windows(padded_h, padded_w, patch_size, stride):
        coords.append((y, x))
        patches.append(padded[y:y + patch_size, x:x + patch_size])

    preds = predict_batch(model, patches, device, batch_size)
    for (y, x), pred in zip(coords, preds):
        for cls in range(N_CLASSES):
            votes[y:y + patch_size, x:x + patch_size, cls] += (pred == cls)

    return votes[:height, :width].argmax(axis=-1).astype(np.uint8)


def read_tiff(path, stats):
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("rasterio is required for GeoTIFF inference.") from exc

    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        meta = src.meta.copy()
    if data.shape[0] < N_CHANNELS:
        raise ValueError(f"{path} has {data.shape[0]} bands; expected at least {N_CHANNELS}.")
    array = np.transpose(data[:N_CHANNELS], (1, 2, 0))
    array = normalize_patch(array, stats)
    return array, meta


def write_tiff_mask(path, mask, meta):
    try:
        import rasterio
    except ImportError:
        return
    out_meta = meta.copy()
    out_meta.update({"count": 1, "dtype": "uint8", "nodata": 255})
    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(mask.astype(np.uint8), 1)


def save_outputs(stem, mask, output_dir, source_array=None, tiff_meta=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{stem}_mask.npy", mask)
    plt.imsave(output_dir / f"{stem}_mask.png", colorize(mask))
    with open(output_dir / f"{stem}_summary.json", "w") as f:
        json.dump(summarize_mask(mask), f, indent=2)

    if tiff_meta is not None:
        write_tiff_mask(output_dir / f"{stem}_mask.tif", mask, tiff_meta)

    if source_array is not None and source_array.shape[-1] >= 3:
        rgb = stretch_rgb(source_array[..., [2, 1, 0]])
        overlay = (0.55 * rgb + 0.45 * (colorize(mask).astype(np.float32) / 255.0))
        plt.imsave(output_dir / f"{stem}_overlay.png", np.clip(overlay, 0, 1))


def collect_inputs(path):
    if path.is_file():
        return [path]
    files = []
    for pattern in ("*.npy", "*.tif", "*.tiff"):
        files.extend(path.glob(pattern))
    return sorted(files)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = load_stats(args.stats)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Device: {device}")

    model = load_checkpoint(args.checkpoint, device)
    inputs = collect_inputs(args.input)
    if not inputs:
        raise RuntimeError(f"No .npy or .tif inputs found at {args.input}")

    run_summary = {}
    for input_path in inputs:
        log.info(f"Predicting {input_path}")
        suffix = input_path.suffix.lower()
        tiff_meta = None
        if suffix == ".npy":
            array = np.load(input_path).astype(np.float32)
        elif suffix in {".tif", ".tiff"}:
            array, tiff_meta = read_tiff(input_path, stats)
        else:
            continue

        mask = predict_array(model, array, device, args.batch_size, args.patch_size, args.stride)
        save_outputs(input_path.stem, mask, args.output_dir, source_array=array, tiff_meta=tiff_meta)
        run_summary[input_path.name] = summarize_mask(mask)

    with open(args.output_dir / "inference_run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    log.info(f"Inference complete: {args.output_dir}")


if __name__ == "__main__":
    main()
