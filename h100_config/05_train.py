"""
05_train.py
Urban Tree Monitoring System — Phase 5: Model Training (H100 GPU)
Full training pipeline with: AMP, gradient accumulation, checkpointing, early stopping.
"""

import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm

# Import our model
from model_04 import UNet, CombinedLoss

# ─── CONFIG ────────────────────────────────────────────────────────────────────

AUGMENTED_DIR  = Path("/Data/username/urban_tree_project/augmented")
MODEL_DIR      = Path("/Data/username/urban_tree_project/models")
RESULTS_DIR    = Path("/Data/username/urban_tree_project/results")
LOG_DIR        = Path("/Data/username/urban_tree_project/logs")

for d in [MODEL_DIR, RESULTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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

# H100 optimization flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True

# ─── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "05_training.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

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

    def __init__(self, data_dir, transform=None):
        self.files     = sorted(Path(data_dir).glob("aug_*.npy"))
        self.transform = transform
        log.info(f"Dataset: {len(self.files)} patches in {data_dir}")

    def __len__(self):
        return len(self.files)

    def generate_pseudo_label(self, patch):
        """
        Generate a 5-class segmentation mask from vegetation indices.
        Replace with real annotation masks when available.
        patch: (H, W, C) — channels indexed per BAND_NAMES
        """
        ndvi = patch[..., 10]   # Band index 10 = NDVI
        bsi  = patch[..., 13]   # Band index 13 = BSI
        blue = patch[..., 0]    # Band index 0  = B2 (Blue)

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

# ─── TRAINING LOOP ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch} [train]")):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with autocast():                     # Mixed precision (FP16 on H100)
            logits = model(x)
            loss   = criterion(logits, y) / GRAD_ACCUM

        scaler.scale(loss).backward()

        # Gradient accumulation
        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou, total_acc = 0.0, 0.0, 0.0

    for x, y in tqdm(loader, desc="Validating"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast():
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
    log.info("=" * 60)
    log.info("Urban Tree Monitoring — Training on H100 GPU")
    log.info("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        log.warning("No GPU found! Training will be very slow.")

    # Dataset + Split
    dataset = VegetationDataset(AUGMENTED_DIR)
    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    log.info(f"Train: {n_train} | Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    # Model
    model     = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
    log.info(f"Parameters: {model.count_parameters():,}")

    # Optimizer + Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    scaler    = GradScaler()

    # Training state
    best_iou       = 0.0
    patience       = 15
    patience_count = 0
    history        = []

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch)
        val_loss, val_iou, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        log.info(
            f"Epoch {epoch:03d}/{EPOCHS} | "
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
            ckpt_path = MODEL_DIR / "best_model.pth"
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_iou':     val_iou,
                'val_acc':     val_acc,
            }, ckpt_path)
            log.info(f"  ✓ Best model saved (mIoU: {best_iou:.4f})")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                log.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

        # Periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict()
            }, MODEL_DIR / f"checkpoint_ep{epoch:03d}.pth")

    # Save history
    elapsed = (time.time() - start_time) / 60
    with open(RESULTS_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    log.info("=" * 60)
    log.info(f"DONE: Best mIoU = {best_iou:.4f} | Total time: {elapsed:.1f} min")
    log.info(f"Model saved at: {MODEL_DIR / 'best_model.pth'}")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
