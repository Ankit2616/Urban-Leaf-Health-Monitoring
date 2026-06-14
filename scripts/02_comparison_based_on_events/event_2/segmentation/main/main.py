# =============================================================================
#  HASDEO FOREST — SEGMENTATION + GIF  (JPG/PNG version)
#  Detects: Dense Forest | Sparse Veg | Barren Land | Mining/Cleared | Shadow
#  GIF order: Pre-event → During → Post-event
# =============================================================================


# ██████████████████████████████████████████████████████████████████████████████
# CELL 1 — INSTALL
# ██████████████████████████████████████████████████████████████████████████████

import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'scikit-image', 'scikit-learn', 'matplotlib',
    'numpy', 'opencv-python-headless', 'Pillow', 'pandas'])

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage import morphology, measure
warnings.filterwarnings('ignore')

print("✅ All packages ready")


# ██████████████████████████████████████████████████████████████████████████████
# CELL 2 — ★ EDIT THIS CELL WITH YOUR FILE NAMES ★
# ██████████████████████████████████████████████████████████████████████████████

# ── Folder where your 5 images are ───────────────────────────────────────────
IMAGE_FOLDER = r'C:\Users\vijay\Desktop\Urban_Tree_Monitoring\Urban-Leaf-Health-Monitoring\scripts\02_comparison_based_on_events\event_2\segmentation\compare_imgs'

# ── Your 5 image filenames IN ORDER (pre → during → post) ────────────────────
#    Just the filename, not full path
IMAGE_FILES = [
    'MON_Hasdeo_North_2022_01_slowdown.jpg',
    'MON_Hasdeo_North_2022_02_slowdown.jpg',
    'MON_Hasdeo_North_2022_03_slowdown.jpg',
    'MON_Hasdeo_North_2022_04_slowdown.jpg',
    'MON_Hasdeo_North_2022_05_slowdown.jpg',
]

# ── Labels shown on plots and GIF frames ─────────────────────────────────────
IMAGE_LABELS = [
    'Jan 2022',
    'Feb 2022',
    'Mar 2022 - Peak Start',
    'Apr 2022 - Peak Event',
    'May 2022',
]

# ── Output folder (created automatically) ─────────────────────────────────────
OUTPUT_DIR = r'C:\Users\vijay\Desktop\Urban_Tree_Monitoring\Urban-Leaf-Health-Monitoring\scripts\02_comparison_based_on_events\event_2\segmentation\output'

# ── GIF settings ──────────────────────────────────────────────────────────────
GIF_DURATION_MS = 1200   # time per frame in ms (increase to slow down)
GIF_SIZE        = (900, 450)   # width x height of each GIF frame

print("✅ Config set")
print(f"   Looking for images in: {IMAGE_FOLDER}")


# ██████████████████████████████████████████████████████████████████████████████
# CELL 3 — LOAD + PREPROCESS IMAGES
# ██████████████████████████████████████████████████████████████████████████████

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

RESIZE_TO = (512, 512)

# 5 land cover classes — designed for your forest + mining use case
N_CLASSES   = 5
CLASS_NAMES = [
    'Dense Forest',       # 0 — high VARI, dark green
    'Sparse Vegetation',  # 1 — moderate VARI, lighter green
    'Barren / Soil',      # 2 — low VARI, brownish
    'Mining / Cleared',   # 3 — very low VARI + high brightness (white/grey)
    'Shadow / Water',     # 4 — very dark
]
CLASS_COLORS = np.array([
    [0.05, 0.45, 0.05],   # 0 Dense forest   — dark green
    [0.55, 0.80, 0.20],   # 1 Sparse veg     — lime green
    [0.75, 0.55, 0.25],   # 2 Barren/soil    — brown
    [0.95, 0.85, 0.50],   # 3 Mining/cleared — pale yellow (bright exposed rock)
    [0.15, 0.20, 0.35],   # 4 Shadow/water   — dark blue-grey
])


def load_jpg_png(filepath):
    """Load jpg/png as normalised float32 RGB (H,W,3) in range 0–1."""
    img = Image.open(filepath).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def resize_rgb(rgb, size=RESIZE_TO):
    img_u8  = (rgb * 255).astype(np.uint8)
    resized = cv2.resize(img_u8, size, interpolation=cv2.INTER_LANCZOS4)
    return resized.astype(np.float32) / 255.0


def compute_indices(rgb):
    """RGB-based vegetation + brightness indices."""
    R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    eps = 1e-7

    # Vegetation greenness
    vari  = np.clip((G - R) / (G + R - B + eps), -1, 1)
    ngrdi = np.clip((G - R) / (G + R + eps), -1, 1)
    exg   = np.clip(2*G - R - B, -1, 1)

    # Brightness — mining areas are very bright (whitish grey)
    brightness = (R + G + B) / 3.0

    # Redness — bare soil tends to be redder than mining rubble
    redness = np.clip((R - G) / (R + G + eps), -1, 1)

    return {
        'VARI'      : vari,
        'NGRDI'     : ngrdi,
        'ExG'       : exg,
        'brightness': brightness,
        'redness'   : redness,
    }


# ── Load all 5 images ─────────────────────────────────────────────────────────
print("Loading images...\n")
images = []
for i, fname in enumerate(IMAGE_FILES):
    fpath = Path(IMAGE_FOLDER) / fname
    if not fpath.exists():
        print(f"  ❌ NOT FOUND — check filename: {fpath}")
        print(f"     Files in folder: {[f.name for f in Path(IMAGE_FOLDER).glob('*.jpg')][:10]}")
        continue
    rgb     = load_jpg_png(fpath)
    rgb_rs  = resize_rgb(rgb)
    indices = compute_indices(rgb_rs)
    images.append({
        'path'   : fpath,
        'label'  : IMAGE_LABELS[i],
        'rgb'    : rgb_rs,
        'idx'    : indices,
        'order'  : i,
    })
    print(f"  ✓ [{i+1}/5] {IMAGE_LABELS[i]:<28}  "
          f"VARI={indices['VARI'].mean():.3f}  "
          f"Bright={indices['brightness'].mean():.3f}")

print(f"\n✅ {len(images)}/5 images loaded")
if len(images) == 0:
    raise SystemExit("\n❌ No images loaded — fix IMAGE_FOLDER and IMAGE_FILES in CELL 2 then rerun")
if len(images) < len(IMAGE_FILES):
    print("\n⚠  Some images missing — continuing with loaded images only")


# ██████████████████████████████████████████████████████████████████████████████
# CELL 4 — SEGMENTATION (K-Means, 5 classes)
# ██████████████████████████████████████████████████████████████████████████████

def segment_image(rgb, idx):
    """
    K-Means on 7 features:
      R, G, B, VARI, NGRDI, brightness, redness
    Classes sorted so:
      0 = densest forest (highest VARI)
      4 = darkest (shadow/water)
    Mining detected by high brightness + low VARI.
    """
    H, W = rgb.shape[:2]

    feat = np.stack([
        rgb[:,:,0],
        rgb[:,:,1],
        rgb[:,:,2],
        idx['VARI'],
        idx['NGRDI'],
        idx['brightness'],
        idx['redness'],
    ], axis=2).reshape(-1, 7)

    valid  = ~np.any(np.isnan(feat) | np.isinf(feat), axis=1)
    feat_v = feat[valid]

    scaler   = StandardScaler()
    feat_s   = scaler.fit_transform(feat_v)

    km = KMeans(n_clusters=N_CLASSES, random_state=42, n_init=15, max_iter=400)
    raw = km.fit_predict(feat_s)

    # Map back to 2D
    flat = np.full(H * W, -1, dtype=int)
    flat[valid] = raw
    label_img   = flat.reshape(H, W)

    # Sort clusters:
    # primary key = mean VARI descending (most green → least green)
    # secondary   = brightness for splitting barren vs mining
    vari_f  = idx['VARI'].ravel()
    brit_f  = idx['brightness'].ravel()

    c_vari  = np.array([vari_f[valid][raw == k].mean()  for k in range(N_CLASSES)])
    c_brit  = np.array([brit_f[valid][raw == k].mean()  for k in range(N_CLASSES)])

    # Sort by VARI descending; break ties by brightness
    sort_key = c_vari - 0.3 * c_brit   # penalise high brightness (mining pushes down)
    order    = np.argsort(sort_key)[::-1]  # 0=most veg, 4=least

    remap     = {old: new for new, old in enumerate(order)}
    sorted_lm = np.vectorize(lambda x: remap.get(x, -1))(label_img)

    # Stats
    total_v = valid.sum()
    stats = {}
    for c in range(N_CLASSES):
        px = (sorted_lm == c).sum()
        stats[c] = {
            'name'  : CLASS_NAMES[c],
            'pixels': px,
            'pct'   : px / total_v * 100 if total_v > 0 else 0,
        }

    return sorted_lm, stats


print("Running segmentation on all images...\n")
for img in images:
    seg, stats = segment_image(img['rgb'], img['idx'])
    img['seg']   = seg
    img['stats'] = stats
    print(f"[{img['label']}]")
    for c in range(N_CLASSES):
        s   = stats[c]
        bar = '█' * int(s['pct'] / 3)
        col = ['🟢','🟡','🟤','⬜','🔵'][c]
        print(f"  {col} {CLASS_NAMES[c]:<22}: {s['pct']:5.1f}%  {bar}")
    print()

print("✅ Segmentation complete")


# ██████████████████████████████████████████████████████████████████████████████
# CELL 5 — SEGMENTATION FIGURE (all 5 images, 3 rows: RGB / Seg / Stats)
# ██████████████████████████████████████████████████████████████████████████████

n   = len(images)
fig = plt.figure(figsize=(n * 3.8, 11), facecolor='#0d1117')
gs  = fig.add_gridspec(3, n, hspace=0.35, wspace=0.08,
                        top=0.93, bottom=0.06, left=0.04, right=0.97)

fig.suptitle(
    'Hasdeo Arand Forest — Land Cover Segmentation\n'
    'Dense Forest  |  Sparse Veg  |  Barren  |  Mining/Cleared  |  Shadow/Water',
    fontsize=13, color='white', fontweight='bold'
)

legend_patches = [
    mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
    for c in range(N_CLASSES)
]

for i, img in enumerate(images):

    # ── Row 0: Original RGB ───────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, i])
    ax0.imshow(img['rgb'])
    ax0.set_title(img['label'], fontsize=8.5, color='white',
                  fontweight='bold', pad=4)
    ax0.axis('off')
    # Phase badge
    phase_map = {0:'PRE', 1:'PRE', 2:'EVENT', 3:'PEAK', 4:'POST'}
    badge_col = {'PRE':'#2196F3','EVENT':'#FF9800','PEAK':'#f44336','POST':'#4CAF50'}
    ph  = phase_map.get(i, '')
    col = badge_col.get(ph, '#555')
    ax0.text(0.02, 0.96, ph, transform=ax0.transAxes,
             fontsize=7, color='white', fontweight='bold',
             bbox=dict(fc=col, ec='none', pad=2, alpha=0.9),
             va='top')

    # ── Row 1: Segmentation map ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, i])
    seg_vis = CLASS_COLORS[np.clip(img['seg'], 0, N_CLASSES-1)]
    ax1.imshow(seg_vis)
    ax1.set_title('Land Cover Map', fontsize=7.5, color='#aaa', pad=3)
    ax1.axis('off')

    # ── Row 2: Horizontal bar chart ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2, i])
    ax2.set_facecolor('#1a1f2e')
    vals  = [img['stats'][c]['pct'] for c in range(N_CLASSES)]
    cols  = [CLASS_COLORS[c] for c in range(N_CLASSES)]
    names = [CLASS_NAMES[c].replace(' / ', '/') for c in range(N_CLASSES)]
    bars  = ax2.barh(names, vals, color=cols, edgecolor='none', height=0.6)
    for bar, v in zip(bars, vals):
        if v > 2:
            ax2.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{v:.1f}%', va='center', fontsize=6.5, color='white')
    ax2.set_xlim(0, 105)
    ax2.tick_params(colors='#ccc', labelsize=6.5)
    ax2.set_xlabel('% area', fontsize=6.5, color='#aaa')
    for sp in ax2.spines.values(): sp.set_visible(False)

fig.legend(handles=legend_patches, loc='lower center', ncol=5,
           fontsize=8, framealpha=0.15, facecolor='#1a1f2e',
           edgecolor='none', labelcolor='white',
           bbox_to_anchor=(0.5, 0.005))

out = f'{OUTPUT_DIR}/01_segmentation_all_phases.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"✅ Saved → {out}")


# ██████████████████████████████████████████████████████████████████████████████
# CELL 6 — CHANGE TREND PLOTS
# ██████████████████████████████████████████████████████████████████████████████

labels      = [img['label'] for img in images]
forest_pct  = [img['stats'][0]['pct'] for img in images]
sparse_pct  = [img['stats'][1]['pct'] for img in images]
barren_pct  = [img['stats'][2]['pct'] for img in images]
mining_pct  = [img['stats'][3]['pct'] for img in images]
shadow_pct  = [img['stats'][4]['pct'] for img in images]
vari_means  = [img['idx']['VARI'].mean() for img in images]
x           = np.arange(len(images))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#0d1117')
fig.suptitle('Hasdeo Forest — Vegetation & Degradation Change  (Pre → Peak → Post)',
             fontsize=13, color='white', fontweight='bold')

# ── Plot 1: Stacked area ──────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#1a1f2e')
ax.stackplot(x, forest_pct, sparse_pct, barren_pct, mining_pct, shadow_pct,
             labels=CLASS_NAMES,
             colors=[CLASS_COLORS[c] for c in range(N_CLASSES)],
             alpha=0.88)
ax.set_xticks(x)
ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=7, color='white')
ax.set_ylabel('% Area', color='white')
ax.set_title('Land Cover Composition Over Time', color='white', fontsize=10)
ax.legend(loc='upper right', fontsize=6.5, facecolor='#1a1f2e',
          edgecolor='none', labelcolor='white')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#444')

# ── Plot 2: Forest vs Mining vs Barren lines ──────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor('#1a1f2e')
ax2.plot(x, forest_pct, 'o-', color='#4CAF50', lw=2.5, ms=8, label='Dense Forest %')
ax2.plot(x, barren_pct, 's-', color='#8D6E63', lw=2.5, ms=8, label='Barren %')
ax2.plot(x, mining_pct, '^-', color='#FFD54F', lw=2.5, ms=8, label='Mining/Cleared %')
for xi, yi in zip(x, forest_pct):
    ax2.text(xi, yi+0.8, f'{yi:.1f}', ha='center', fontsize=7, color='#4CAF50')
for xi, yi in zip(x, mining_pct):
    ax2.text(xi, yi+0.8, f'{yi:.1f}', ha='center', fontsize=7, color='#FFD54F')
# Shade peak event
ax2.axvspan(2.8, 3.2, alpha=0.25, color='red', label='Peak event')
ax2.set_xticks(x)
ax2.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=7, color='white')
ax2.set_ylabel('% Area', color='white')
ax2.set_title('Forest vs Mining vs Barren', color='white', fontsize=10)
ax2.legend(fontsize=7, facecolor='#1a1f2e', edgecolor='none', labelcolor='white')
ax2.tick_params(colors='white')
for sp in ax2.spines.values(): sp.set_color('#444')

# ── Plot 3: VARI vegetation index ─────────────────────────────────────────────
ax3 = axes[2]
ax3.set_facecolor('#1a1f2e')
bar_colors = ['#2196F3','#2196F3','#FF9800','#f44336','#4CAF50']
bars = ax3.bar(x, vari_means, color=bar_colors[:len(images)], edgecolor='none', width=0.6)
for bar, v in zip(bars, vari_means):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 0.002,
             f'{v:.4f}', ha='center', va='bottom', fontsize=8, color='white')
ax3.plot(x, vari_means, 'o--', color='white', lw=1.5, ms=5, alpha=0.6)
ax3.set_xticks(x)
ax3.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=7, color='white')
ax3.set_ylabel('Mean VARI', color='white')
ax3.set_title('VARI Vegetation Index\n(higher = more green)', color='white', fontsize=10)
ax3.tick_params(colors='white')
for sp in ax3.spines.values(): sp.set_color('#444')

plt.tight_layout()
out = f'{OUTPUT_DIR}/02_change_trends.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"✅ Saved → {out}")


# ██████████████████████████████████████████████████████████████████████████████
# CELL 7 — BUILD AND SAVE 3 GIFs
#   gif_01_rgb.gif        — raw RGB timeline
#   gif_02_segmented.gif  — segmentation maps
#   gif_03_combined.gif   — RGB + seg side by side  ← best one
# ██████████████████████████████████████████████████████████████████████████████

PHASE_BADGES = {
    0: ('JAN 2022',        '#FF9800'),
    1: ('FEB 2022',        '#FF9800'),
    2: ('MAR 2022 - PEAK', '#f44336'),
    3: ('APR 2022 - PEAK', '#f44336'),
    4: ('MAY 2022',        '#FF9800'),
}

def arr_to_pil(arr_float, size):
    """Float32 RGB (0-1) → PIL image resized to size (W,H)."""
    u8 = (np.clip(arr_float, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(u8).resize(size, Image.LANCZOS)


def draw_label(draw, text, xy, fill=(255,255,255), bg=None, fontsize=14):
    """Draw text with optional background box."""
    x, y = xy
    if bg:
        w = len(text) * (fontsize * 0.6)
        draw.rectangle([x-4, y-3, x+w+4, y+fontsize+3], fill=bg)
    draw.text((x, y), text, fill=fill)


def make_combined_frame(img, order_idx, frame_size=(900, 450)):
    """One GIF frame: left=RGB, right=segmentation, with overlays."""
    W, H   = frame_size
    half   = W // 2
    pad    = 2

    # Left: RGB
    rgb_pil = arr_to_pil(img['rgb'], (half - pad, H - 40))
    # Right: segmentation
    seg_arr = CLASS_COLORS[np.clip(img['seg'], 0, N_CLASSES-1)]
    seg_pil = arr_to_pil(seg_arr, (half - pad, H - 40))

    # Canvas
    canvas = Image.new('RGB', (W, H), (13, 17, 23))

    # Header bar
    header = Image.new('RGB', (W, 40), (20, 26, 48))
    canvas.paste(header, (0, 0))

    # Paste images
    canvas.paste(rgb_pil, (0, 40))
    canvas.paste(seg_pil, (half + pad, 40))

    draw = ImageDraw.Draw(canvas)

    # Vertical divider
    draw.line([(half, 0), (half, H)], fill=(60, 70, 100), width=2)

    # Header text
    badge_text, badge_col = PHASE_BADGES.get(order_idx, ('', '#555'))
    draw.rectangle([8, 8, 110, 32], fill=badge_col)
    draw.text((12, 10), badge_text, fill='white')
    draw.text((120, 10), img['label'], fill=(220, 220, 220))
    draw.text((half + 10, 10), 'Land Cover Segmentation', fill=(180, 180, 200))

    # Panel labels
    draw.text((6, 44), 'Original Image', fill=(150, 180, 150))
    draw.text((half + 6, 44), 'Classified Map', fill=(150, 180, 210))

    # Stats overlay bottom-left
    stats_y = H - (N_CLASSES * 17) - 8
    draw.rectangle([0, stats_y - 4, 195, H], fill=(10, 12, 20))
    for c in range(N_CLASSES):
        col_box = tuple((CLASS_COLORS[c] * 255).astype(int))
        pct     = img['stats'][c]['pct']
        draw.rectangle([6, stats_y, 16, stats_y+12], fill=col_box)
        draw.text((20, stats_y - 1), f"{CLASS_NAMES[c][:16]}: {pct:.1f}%",
                  fill=(210, 210, 210))
        stats_y += 17

    # Legend bottom-right (seg panel)
    leg_x = half + 8
    leg_y = H - (N_CLASSES * 17) - 8
    draw.rectangle([leg_x - 4, leg_y - 4, W - 2, H], fill=(10, 12, 20))
    for c in range(N_CLASSES):
        col_box = tuple((CLASS_COLORS[c] * 255).astype(int))
        draw.rectangle([leg_x, leg_y, leg_x+14, leg_y+12], fill=col_box)
        draw.text((leg_x + 18, leg_y - 1), CLASS_NAMES[c], fill=(210, 210, 210))
        leg_y += 17

    return canvas


def make_rgb_frame(img, order_idx, size=(900, 450)):
    canvas = Image.new('RGB', size, (13, 17, 23))
    rgb_pil = arr_to_pil(img['rgb'], (size[0], size[1] - 40))
    canvas.paste(rgb_pil, (0, 40))
    header = Image.new('RGB', (size[0], 40), (20, 26, 48))
    canvas.paste(header, (0, 0))
    draw = ImageDraw.Draw(canvas)
    badge_text, badge_col = PHASE_BADGES.get(order_idx, ('', '#555'))
    draw.rectangle([8, 8, 110, 32], fill=badge_col)
    draw.text((12, 10), badge_text, fill='white')
    draw.text((120, 10), img['label'], fill=(220, 220, 220))
    return canvas


def make_seg_frame(img, order_idx, size=(900, 450)):
    canvas = Image.new('RGB', size, (13, 17, 23))
    seg_arr = CLASS_COLORS[np.clip(img['seg'], 0, N_CLASSES-1)]
    seg_pil = arr_to_pil(seg_arr, (size[0], size[1] - 40))
    canvas.paste(seg_pil, (0, 40))
    header = Image.new('RGB', (size[0], 40), (20, 26, 48))
    canvas.paste(header, (0, 0))
    draw = ImageDraw.Draw(canvas)
    badge_text, badge_col = PHASE_BADGES.get(order_idx, ('', '#555'))
    draw.rectangle([8, 8, 110, 32], fill=badge_col)
    draw.text((12, 10), badge_text, fill='white')
    draw.text((120, 10), f"{img['label']} — Segmented", fill=(220, 220, 220))
    return canvas


# ── Build frames ──────────────────────────────────────────────────────────────
print("Building GIF frames...\n")
frames_combined = []
frames_rgb      = []
frames_seg      = []

for i, img in enumerate(images):
    frames_combined.append(make_combined_frame(img, i, GIF_SIZE))
    frames_rgb.append(make_rgb_frame(img, i, GIF_SIZE))
    frames_seg.append(make_seg_frame(img, i, GIF_SIZE))
    print(f"  ✓ Frame {i+1}/5: {img['label']}")

# ── Save GIFs ─────────────────────────────────────────────────────────────────
def save_gif(frames, path, duration=GIF_DURATION_MS):
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=duration, loop=0, optimize=False,
    )
    kb = Path(path).stat().st_size // 1024
    print(f"  ✅ {Path(path).name:<35} {len(frames)} frames  {kb} KB")

print("\nSaving GIFs...")
save_gif(frames_combined, f'{OUTPUT_DIR}/gif_03_combined.gif')
save_gif(frames_rgb,      f'{OUTPUT_DIR}/gif_01_rgb.gif',       duration=1000)
save_gif(frames_seg,      f'{OUTPUT_DIR}/gif_02_segmented.gif', duration=1000)

print(f"\n📁 All files at: {OUTPUT_DIR}")


# ██████████████████████████████████████████████████████████████████████████████
# CELL 8 — PREVIEW GIFs INLINE
# ██████████████████████████████████████████████████████████████████████████████

from IPython.display import Image as IPImage, display, HTML

display(HTML("<h3 style='color:white;background:#0d1117;padding:8px'>📽 Combined GIF Preview</h3>"))
display(IPImage(filename=f'{OUTPUT_DIR}/gif_03_combined.gif', width=700))

display(HTML("<h3 style='color:white;background:#0d1117;padding:8px'>🗺 Segmentation GIF</h3>"))
display(IPImage(filename=f'{OUTPUT_DIR}/gif_02_segmented.gif', width=700))


# ██████████████████████████████████████████████████████████████████████████████
# CELL 9 — SUMMARY TABLE
# ██████████████████████████████████████████████████████████████████████████████

rows = []
for img in images:
    row = {'Phase': img['label']}
    for c in range(N_CLASSES):
        row[CLASS_NAMES[c] + ' %'] = round(img['stats'][c]['pct'], 1)
    row['VARI mean'] = round(img['idx']['VARI'].mean(), 4)
    rows.append(row)

df = pd.DataFrame(rows)
print("── SEGMENTATION SUMMARY ─────────────────────────────────────────────────")
print(df.to_string(index=False))

df.to_csv(f'{OUTPUT_DIR}/03_summary.csv', index=False)
print(f"\n✅ CSV saved → {OUTPUT_DIR}/03_summary.csv")

# ── Key change summary ────────────────────────────────────────────────────────
first, last = images[0], images[-1]
peak        = images[3]
print(f"\n── KEY CHANGES ──────────────────────────────────────────────────────────")
print(f"  {first['label']}  →  {last['label']}")
print(f"  Dense Forest  : {first['stats'][0]['pct']:.1f}% → {last['stats'][0]['pct']:.1f}%   Δ {last['stats'][0]['pct']-first['stats'][0]['pct']:+.1f}%")
print(f"  Barren/Soil   : {first['stats'][2]['pct']:.1f}% → {last['stats'][2]['pct']:.1f}%   Δ {last['stats'][2]['pct']-first['stats'][2]['pct']:+.1f}%")
print(f"  Mining/Cleared: {first['stats'][3]['pct']:.1f}% → {last['stats'][3]['pct']:.1f}%   Δ {last['stats'][3]['pct']-first['stats'][3]['pct']:+.1f}%")
print(f"\n  Peak Event ({peak['label']}):")
print(f"  Dense Forest  : {peak['stats'][0]['pct']:.1f}%")
print(f"  Mining/Cleared: {peak['stats'][3]['pct']:.1f}%")
print(f"  VARI          : {peak['idx']['VARI'].mean():.4f}")
