"""
05_train.py
Urban Tree Monitoring System — Phase 5: Model Training (H100 GPU)
Full training pipeline with: AMP, gradient accumulation, checkpointing, early stopping.
"""

import argparse
import csv
import importlib.util
import os
import time
import json
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm

def load_model_module():
    """Load 04_model.py without relying on an importable numeric module name."""
    module_path = Path(__file__).with_name("04_model.py")
    spec = importlib.util.spec_from_file_location("urban_leaf_model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_model_module = load_model_module()
UNet = _model_module.UNet
CombinedLoss = _model_module.CombinedLoss

# ─── CONFIG ────────────────────────────────────────────────────────────────────

AUGMENTED_DIR  = Path("/Data/username/urban_tree_project/augmented")
MODEL_DIR      = Path("/Data/username/urban_tree_project/models")
RESULTS_DIR    = Path("/Data/username/urban_tree_project/results")
LOG_DIR        = Path("/Data/username/urban_tree_project/logs")

# Model
N_CHANNELS   = 14
N_CLASSES    = 5
IMG_SIZE     = 256

# Training
EPOCHS       = 100
BATCH_SIZE   = 16     # Fits well on H100 40GB MIG slice
LR           = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_ACCUM   = 2      # Effective batch = 32
VAL_SPLIT    = 0.15
NUM_WORKERS  = 8      # H100 node has many CPU cores available
SEED         = 42

# H100 optimization flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True

# ─── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Attention U-Net for vegetation segmentation.")
    parser.add_argument("--data-dir", type=Path, default=AUGMENTED_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")
    return parser.parse_args()


def configure_file_logging(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "05_training.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ─── DATASET ───────────────────────────────────────────────────────────────────

class VegetationDataset(Dataset):
    """
    Loads augmented satellite patches and generates pseudo-labels from NDVI
    for unsupervised/self-supervised pre-training.

    In production, replace generate_pseudo_label() with your actual annotation masks.
    Supported label formats: GeoTIFF mask, .npy mask, or shapefile-rasterized labels.
    """
    CLASS_MAP = {
        0: "Vegetation",        # NDVI > 0.4
        1: "Sparse Vegetation", # 0.2 < NDVI ≤ 0.4
        2: "Bare Soil/Rock",    # -0.1 < NDVI ≤ 0.2
        3: "Built-up/Urban",    # NDVI ≤ -0.1
        4: "Water/Shadow",      # BSI threshold
    }

    DEFAULT_STATS = {
        "B2": (0.0500, 0.0300),
        "NDVI": (0.4000, 0.2000),
        "BSI": (-0.1000, 0.2000),
    }

    def __init__(self, data_dir, transform=None):
        data_dir = Path(data_dir)
        self.files = sorted(data_dir.glob("aug_*.npy"))
        if not self.files:
            self.files = sorted(data_dir.glob("patch_*.npy"))
        self.transform = transform
        self.stats = self._load_stats(data_dir)
        log.info(f"Dataset: {len(self.files)} patches in {data_dir}")

    def __len__(self):
        return len(self.files)

    def _load_stats(self, data_dir):
        stats_path = data_dir.parent / "processed" / "band_statistics.json"
        if not stats_path.exists():
            return self.DEFAULT_STATS
        try:
            with open(stats_path) as f:
                raw = json.load(f)
            return {k: tuple(v) for k, v in raw.items()}
        except Exception as exc:
            log.warning(f"Could not load {stats_path}: {exc}. Using default pseudo-label stats.")
            return self.DEFAULT_STATS

    def _denormalize(self, values, band_name):
        mean, std = self.stats.get(band_name, self.DEFAULT_STATS[band_name])
        return values * std + mean

    def generate_pseudo_label(self, patch):
        """
        Generate a 5-class segmentation mask from vegetation indices.
        Replace with real annotation masks when available.
        patch: (H, W, C) — channels indexed per BAND_NAMES
        """
        ndvi = self._denormalize(patch[..., 10], "NDVI")
        bsi  = self._denormalize(patch[..., 13], "BSI")
        blue = self._denormalize(patch[..., 0], "B2")

        label = np.zeros((patch.shape[0], patch.shape[1]), dtype=np.int64)
        label[ndvi > 0.4]                           = 0  # Dense vegetation
        label[(ndvi > 0.2) & (ndvi <= 0.4)]         = 1  # Sparse vegetation
        label[(ndvi > -0.1) & (ndvi <= 0.2)]        = 2  # Soil/rock
        label[ndvi <= -0.1]                         = 3  # Built-up
        label[(blue > 0.05) & (ndvi < 0.0) & (bsi < -0.2)] = 4  # Water/shadow
        return label

    def __getitem__(self, idx):
        patch = np.load(self.files[idx]).astype(np.float32)  # (H, W, C)
        label = self.generate_pseudo_label(patch)             # (H, W)

        # Convert to tensors: (C, H, W) for PyTorch
        x = torch.from_numpy(patch.transpose(2, 0, 1))  # (C, H, W)
        y = torch.from_numpy(label)                       # (H, W) int64
        return x, y

# ─── METRICS ───────────────────────────────────────────────────────────────────

def compute_iou(pred, target, n_classes=N_CLASSES):
    """Mean Intersection-over-Union."""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(n_classes):
        pred_mask   = pred == cls
        target_mask = target == cls
        intersection = (pred_mask & target_mask).sum().float()
        union        = (pred_mask | target_mask).sum().float()
        if union == 0:
            continue
        ious.append((intersection / union).item())
    return np.mean(ious) if ious else 0.0

def compute_pixel_acc(pred, target):
    """Pixel accuracy."""
    correct = (pred == target).sum().float()
    total   = target.numel()
    return (correct / total).item()


def class_distribution(dataset, n_classes=N_CLASSES):
    counts = np.zeros(n_classes, dtype=np.int64)
    for file_path in tqdm(dataset.files, desc="Scanning class distribution"):
        patch = np.load(file_path).astype(np.float32)
        labels = dataset.generate_pseudo_label(patch)
        counts += np.bincount(labels.reshape(-1), minlength=n_classes)[:n_classes]
    total = counts.sum()
    return {
        str(cls): {
            "name": VegetationDataset.CLASS_MAP[cls],
            "pixels": int(counts[cls]),
            "ratio": float(counts[cls] / total) if total else 0.0,
        }
        for cls in range(n_classes)
    }


def write_history_csv(history, path):
    if not history:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

# ─── TRAINING LOOP ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch} [train]")):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
            loss   = criterion(logits, y) / grad_accum

        scaler.scale(loss).backward()

        # Gradient accumulation
        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou, total_acc = 0.0, 0.0, 0.0

    for x, y in tqdm(loader, desc="Validating"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(x)
            loss   = criterion(logits, y)

        preds = logits.argmax(dim=1)
        total_loss += loss.item()
        total_iou  += compute_iou(preds, y)
        total_acc  += compute_pixel_acc(preds, y)

    n = len(loader)
    return total_loss / n, total_iou / n, total_acc / n

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    for d in [args.model_dir, args.results_dir, args.log_dir]:
        d.mkdir(parents=True, exist_ok=True)
    configure_file_logging(args.log_dir)
    seed_everything(args.seed)

    log.info("=" * 60)
    log.info("Urban Tree Monitoring — Training on H100 GPU")
    log.info("=" * 60)
    log.info(f"Data dir: {args.data_dir}")
    log.info(f"Results dir: {args.results_dir}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        log.warning("No GPU found! Training will be very slow.")

    # Dataset + Split
    dataset = VegetationDataset(args.data_dir)
    if len(dataset) < 2:
        raise RuntimeError(
            f"Need at least 2 training patches in {args.data_dir}. "
            "Run preprocessing and augmentation first."
        )

    distribution = class_distribution(dataset)
    with open(args.results_dir / "pseudo_label_distribution.json", "w") as f:
        json.dump(distribution, f, indent=2)

    n_val   = max(1, int(len(dataset) * args.val_split))
    if n_val >= len(dataset):
        n_val = len(dataset) - 1
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(args.seed))

    log.info(f"Train: {n_train} | Val: {n_val}")

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # Model
    model     = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
    log.info(f"Parameters: {model.count_parameters():,}")

    # Optimizer + Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    scaler    = GradScaler(device.type, enabled=device.type == "cuda")

    # Training state
    best_iou       = -1.0
    start_epoch    = 1
    patience_count = 0
    history        = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optim_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optim_state"])
        best_iou = float(checkpoint.get("val_iou", best_iou))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        log.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, args.grad_accum)
        val_loss, val_iou, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        log.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {val_iou:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {lr_now:.2e}"
        )

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'val_iou': val_iou,
            'val_acc': val_acc, 'lr': lr_now
        })

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            ckpt_path = args.model_dir / "best_model.pth"
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_iou':     val_iou,
                'val_acc':     val_acc,
                'class_map':   VegetationDataset.CLASS_MAP,
                'n_channels':  N_CHANNELS,
                'n_classes':   N_CLASSES,
            }, ckpt_path)
            log.info(f"  ✓ Best model saved (mIoU: {best_iou:.4f})")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log.info(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs).")
                break

        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict()
            }, args.model_dir / f"checkpoint_ep{epoch:03d}.pth")

    # Save history
    elapsed = (time.time() - start_time) / 60
    with open(args.results_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    write_history_csv(history, args.results_dir / "training_history.csv")

    log.info("=" * 60)
    log.info(f"DONE: Best mIoU = {best_iou:.4f} | Total time: {elapsed:.1f} min")
    log.info(f"Model saved at: {args.model_dir / 'best_model.pth'}")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
