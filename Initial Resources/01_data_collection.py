"""
01_data_collection.py
Urban Tree Monitoring System — Phase 1: Data Collection
Region: Hasdeo Forest (Region #1)
Downloads high-quality multi-band TIFF images from Sentinel-2 via Google Earth Engine.
Target: 1000+ raw images after cloud masking.
"""

import ee
import os
import time
import json
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# ─── CONFIG ────────────────────────────────────────────────────────────────────

GEE_PROJECT_ID = "your-gee-project-id"   # ← Replace with your GEE project ID

# Hasdeo Forest, Chhattisgarh, India
HASDEO_COORDS = [
    82.00, 22.50,   # min lon, min lat
    83.00, 23.50    # max lon, max lat
]

OUTPUT_DIR    = Path("/Data/username/urban_tree_project/raw_tiff")  # HPC path
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR       = Path("/Data/username/urban_tree_project/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Study period — long range for temporal analysis
START_DATE    = "2018-01-01"
END_DATE      = "2024-12-31"
CLOUD_PERCENT = 10        # Strict cloud filter for vegetation studies
MAX_IMAGES    = 1200      # Collect 1200 → aim for 1000+ after QC
SCALE         = 10        # 10m Sentinel-2 native resolution

# Bands to download (RGB + NIR + SWIR + Red-Edge for full feature set)
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# ─── LOGGING ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "01_data_collection.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── EARTH ENGINE INIT ─────────────────────────────────────────────────────────

def init_gee():
    log.info("Initializing Google Earth Engine...")
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        ee.Number(1).add(1).getInfo()  # quick connectivity test
        log.info("GEE initialized successfully.")
    except Exception as e:
        log.warning(f"Auth required: {e}. Starting authentication...")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)
        log.info("GEE authenticated and initialized.")

# ─── CLOUD MASKING ─────────────────────────────────────────────────────────────

def mask_s2_clouds(image):
    """
    Mask clouds and cirrus using QA60 band.
    Scales reflectance to 0-1 and adds acquisition date.
    """
    qa = image.select('QA60')
    cloud_mask  = qa.bitwiseAnd(1 << 10).eq(0)
    cirrus_mask = qa.bitwiseAnd(1 << 11).eq(0)
    mask = cloud_mask.And(cirrus_mask)

    return (
        image.updateMask(mask)
             .divide(10000)
             .select(BANDS)
             .copyProperties(image, ['system:time_start', 'CLOUDY_PIXEL_PERCENTAGE'])
    )

# ─── VEGETATION INDICES ────────────────────────────────────────────────────────

def add_indices(image):
    """Add NDVI, EVI, SAVI, BSI as additional bands."""
    # NDVI: vegetation density
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # EVI: Enhanced Vegetation Index (less saturation in dense canopy)
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': image.select('B8'), 'RED': image.select('B4'), 'BLUE': image.select('B2')}
    ).rename('EVI')

    # SAVI: Soil Adjusted Vegetation Index (L=0.5)
    savi = image.expression(
        '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
        {'NIR': image.select('B8'), 'RED': image.select('B4')}
    ).rename('SAVI')

    # BSI: Bare Soil Index
    bsi = image.expression(
        '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
        {
            'SWIR': image.select('B11'), 'RED': image.select('B4'),
            'NIR':  image.select('B8'),  'BLUE': image.select('B2')
        }
    ).rename('BSI')

    return image.addBands([ndvi, evi, savi, bsi])

# ─── BUILD COLLECTION ──────────────────────────────────────────────────────────

def build_collection(roi):
    log.info(f"Building Sentinel-2 collection: {START_DATE} → {END_DATE}, cloud < {CLOUD_PERCENT}%")

    s2 = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(START_DATE, END_DATE)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_PERCENT))
        .map(mask_s2_clouds)
        .map(add_indices)
        .sort('system:time_start')
    )

    total = s2.size().getInfo()
    log.info(f"Total clean images available: {total}")

    if total < 100:
        log.warning("Low image count! Consider relaxing cloud threshold.")

    return s2.limit(MAX_IMAGES)

# ─── EXPORT TIFF via GETDOWNLOADURL ────────────────────────────────────────────

def download_tiff(image, index, roi, output_dir, retries=3):
    """
    Downloads a multi-band GeoTIFF image using getDownloadURL.
    Includes NDVI, EVI, SAVI, BSI bands for full ML feature set.
    """
    try:
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        cloud = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        cloud_str = f"{float(cloud):.1f}" if cloud is not None else "NA"
    except Exception as e:
        log.error(f"[{index:04d}] Metadata fetch failed: {e}")
        return False

    filename = output_dir / f"hasdeo_{index:04d}_{date}_cloud{cloud_str}.tif"

    if filename.exists():
        log.info(f"[{index:04d}] Already exists: {filename.name} — skipping.")
        return True

    # All bands including indices
    all_bands = BANDS + ['NDVI', 'EVI', 'SAVI', 'BSI']

    for attempt in range(1, retries + 1):
        try:
            url = image.select(all_bands).getDownloadURL({
                'region':     roi,
                'scale':      SCALE,
                'format':     'GEO_TIFF',
                'crs':        'EPSG:4326',
                'filePerBand': False   # Single multi-band TIFF
            })

            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()

            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_mb = filename.stat().st_size / (1024 * 1024)
            log.info(f"[{index:04d}] ✓ {filename.name} ({size_mb:.1f} MB)")
            return True

        except Exception as e:
            log.warning(f"[{index:04d}] Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)  # Exponential backoff

    log.error(f"[{index:04d}] ✗ All attempts failed for {date}")
    return False

# ─── SAVE METADATA ─────────────────────────────────────────────────────────────

def save_metadata(image_list_info, output_dir):
    metadata = []
    for i, feat in enumerate(image_list_info):
        props = feat.get('properties', {})
        metadata.append({
            'index':  i,
            'date':   props.get('date', 'unknown'),
            'cloud':  props.get('CLOUDY_PIXEL_PERCENTAGE', None),
        })

    meta_path = output_dir.parent / "metadata" / "collection_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Metadata saved: {meta_path}")

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Urban Tree Monitoring — Data Collection (Hasdeo Forest)")
    log.info("=" * 60)

    init_gee()

    roi = ee.Geometry.Rectangle(HASDEO_COORDS)
    area_km2 = roi.area().divide(1e6).getInfo()
    log.info(f"ROI area: {area_km2:.1f} km²")

    collection = build_collection(roi)
    count = collection.size().getInfo()
    log.info(f"Downloading {count} images to {OUTPUT_DIR}")

    image_list = collection.toList(count)

    success, failed = 0, 0
    start_time = time.time()

    for i in range(count):
        img = ee.Image(image_list.get(i))
        ok = download_tiff(img, i, roi, OUTPUT_DIR)
        if ok:
            success += 1
        else:
            failed += 1
        time.sleep(0.5)  # Be polite to GEE API

    elapsed = (time.time() - start_time) / 60
    log.info("=" * 60)
    log.info(f"DONE: {success} downloaded, {failed} failed | Time: {elapsed:.1f} min")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info("=" * 60)

if __name__ == "__main__":
    main()
