"""
04_model.py
Urban Tree Monitoring System — Phase 4: U-Net Model Architecture
Custom U-Net with multi-channel satellite input (14 bands).
Supports segmentation into: Vegetation, Buildings, Soil, Water, Urban Expansion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── CONFIG ────────────────────────────────────────────────────────────────────

N_CHANNELS  = 14   # Input bands (optical + vegetation indices)
N_CLASSES   = 5    # Vegetation | Buildings | Soil | Water | Urban Expansion
IMG_SIZE    = 256

# ─── BUILDING BLOCKS ───────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv → BN → ReLU blocks."""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downsample with MaxPool then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample then DoubleConv with skip connection."""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad to handle odd input sizes
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x  = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention gate to focus on relevant feature regions.
    Improves segmentation in dense urban/forest scenes.
    """
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, 1, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, 1, bias=True),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Align spatial dimensions
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ─── U-NET MODEL ───────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Attention U-Net for multi-spectral satellite image segmentation.
    Input:  (B, 14, 256, 256) — 14-band satellite patches
    Output: (B, N_CLASSES, 256, 256) — per-pixel class logits
    """
    def __init__(self, n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        f = [64, 128, 256, 512, 1024]

        # Encoder
        self.inc   = DoubleConv(n_channels, f[0])
        self.down1 = Down(f[0], f[1])
        self.down2 = Down(f[1], f[2])
        self.down3 = Down(f[2], f[3])
        factor     = 2 if bilinear else 1
        self.down4 = Down(f[3], f[4] // factor)

        # Attention gates
        self.att3  = AttentionGate(f[3] // factor, f[3], f[3] // 2)
        self.att2  = AttentionGate(f[2] // factor, f[2], f[2] // 2)
        self.att1  = AttentionGate(f[1] // factor, f[1], f[1] // 2)

        # Decoder
        self.up1   = Up(f[4], f[3] // factor, bilinear)
        self.up2   = Up(f[3], f[2] // factor, bilinear)
        self.up3   = Up(f[2], f[1] // factor, bilinear)
        self.up4   = Up(f[1], f[0], bilinear)

        # Output
        self.outc  = nn.Conv2d(f[0], n_classes, kernel_size=1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with attention
        x4a = self.att3(x5, x4)
        out = self.up1(x5, x4a)

        x3a = self.att2(out, x3)
        out = self.up2(out, x3a)

        x2a = self.att1(out, x2)
        out = self.up3(out, x2a)
        out = self.up4(out, x1)

        return self.outc(out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ─── LOSS FUNCTIONS ────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Dice loss for handling class imbalance in segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs  = torch.softmax(logits, dim=1)
        n_cls  = logits.shape[1]
        dice   = 0.0
        for c in range(n_cls):
            pred_c   = probs[:, c]
            target_c = (targets == c).float()
            num   = 2 * (pred_c * target_c).sum()
            denom = pred_c.sum() + target_c.sum() + self.smooth
            dice += (num / denom)
        return 1 - (dice / n_cls)


class CombinedLoss(nn.Module):
    """
    CE + Dice combined loss.
    CE handles per-pixel accuracy; Dice handles class imbalance.
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_w    = ce_weight
        self.dice_w  = dice_weight
        self.ce      = nn.CrossEntropyLoss()
        self.dice    = DiceLoss()

    def forward(self, logits, targets):
        ce_loss   = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_w * ce_loss + self.dice_w * dice_loss

# ─── QUICK TEST ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model  = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    params = model.count_parameters()
    print(f"Model parameters: {params:,}")

    # Dummy forward pass
    x      = torch.randn(2, N_CHANNELS, IMG_SIZE, IMG_SIZE)
    out    = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print("Model test passed ✓")
