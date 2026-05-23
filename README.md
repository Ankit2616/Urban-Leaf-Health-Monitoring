# Urban-Leaf-Health-Monitoring

Urban-Leaf-Health-Monitoring is an end-to-end system for tracking urban vegetation health and land-use change. Leveraging satellite and aerial imagery, machine learning, and computer vision, it enables robust monitoring of ecological shifts, urban expansion, and deforestation events across diverse regions.

## Key Insights

- **Automated Area Selection:** Efficient sampling and region-of-interest extraction for large-scale analysis.
- **Cloud-Masked Data Pipeline:** Preprocessing ensures high-quality, cloud-free imagery for reliable results.
- **Augmentation & Restoration:** Extensive data augmentation (flipping, zooming, restoration) boosts model robustness.
- **Multispectral Analysis:** Radiometric normalization and spectral transformations (NDVI, EVI, SAVI) provide deep vegetation health insights.
- **Feature Engineering:** Texture (GLCM), spectral statistics, and morphological features enhance classification accuracy.
- **Flexible Modeling:** Supports SVM, Random Forest, and U-Net for both segmentation and health classification.
- **Temporal Event Tracking:** Enables comparison of ecological events (e.g., bushfires, mining) and multi-year change detection.
- **Urban Metrics Visualization:** Generates actionable insights for city planning and conservation.


## Study Regions

- **Hasdeo Forest:** Deforestation and mining impacts (2018–2023)
- **Sydney Blue Mountains Fringe:** Urban expansion and ecological shifts
- **Kangaroo Island:** Black Summer bushfire impacts

## Repository Structure

```
.
├── assets/
│   ├── plots/                  # Visualizations and result plots
│   └── presentation_images/    # Key images for presentations
├── data/
│   ├── 01_area_of_interest_selection_using_sampling/
│   │   ├── batch_1/            # Hasdeo Forest dataset
│   │   ├── batch_2/            # Sydney Blue Mountains dataset
│   │   └── batch_3/            # Kangaroo Island dataset
│   └── 02_comparison_based_on_events/
│       ├── event_1/            # Hasdeo event CSVs and logs
│       └── event_2/            # Additional event CSVs
├── h100_config/                # HPC scripts, configs, and guides
├── initial_resources/          # Reference calculations
├── scripts/
│   ├── 01_area_of_interest_selection/
│   ├── 02_comparison_based_on_events/
│   ├── 03_comparison_based_on_years/
│   └── sample/
├── LICENSE
├── README.md
```

## Features & Roadmap

### Phase 1: Data Engineering

- [ ] Data Collection: 1,000+ raw images (cloud-masked)
- [ ] Augmentation: 5,000+ samples via flipping, zooming, restoration
- [ ] Preprocessing: Radiometric normalization, scaling, multispectral transformation

### Phase 2: Core Analytics

- [ ] Feature Engineering: GLCM texture, spectral stats, morphology
- [ ] Spectral Indices: NDVI, EVI, SAVI
- [ ] Modeling: SVM, Random Forest, U-Net

### Phase 3: Temporal & Event Analysis

- [ ] Event Comparison: March–April 2022 Hasdeo events
- [ ] Time-Lapse Visualization: 5-year change detection
- [ ] Urban Metrics: Feature visualization for city planning

## Quick Start

### Prerequisites

- Python 3.9+
- Spatial data libraries (Rasterio, GDAL)
- PyTorch / TensorFlow

### Installation

```bash
git clone https://github.com/Manjushwarofficial/Urban-Leaf-Health-Monitoring.git
cd Urban-Leaf-Health-Monitoring
pip install -r requirements.txt
```

## Evaluation Metrics

- **Segmentation:** Mean Intersection over Union (mIoU), Dice Coefficient
- **Classification:** Precision-Recall, Confusion Matrix
- **Temporal:** Quantitative land-cover loss


## Contributing

Contributions are welcome! Help select new regions or improve the augmentation pipeline by opening an issue.

## License

MIT License - see [LICENSE](https://github.com/Manjushwarofficial/Urban-Leaf-Health-Monitoring/blob/main/LICENSE)