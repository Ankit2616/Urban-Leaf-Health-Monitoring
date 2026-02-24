
# Urban-Leaf-Health-Monitoring

 A satellite and aerial imagery system designed to segment urban vegetation, compute health indices, and classify forest degradation using Machine Learning and Computer Vision.


## Project Overview

This project monitors vegetation health and land-use changes with a specific focus on high-risk ecological zones and urban expansion. By leveraging multispectral data, we provide a pipeline for:

* **Semantic Segmentation:** Identifying green cover vs. urban sprawl.
* **Health Classification:** Distinguishing between healthy forest, degraded land, and cleared areas using SVM, Random Forest, and CNNs.
* **Temporal Analysis:** Tracking deforestation events (e.g., Hasdeo Forest) and long-term urban shifts.


## Study Regions

1. **Hasdeo Forest:** Focused on event-based deforestation and mining impacts (2018–2023).
2. **[Region #2 - TBD]:** Focused on long-term urban heat island effects or city planning changes.



## Features & Roadmap

### Phase 1: Data Engineering

* [ ] **Data Collection:** Acquisition of 1,000+ high-quality raw images (post-cloud masking).
* [ ] **Augmentation:** Scaling dataset to 5,000+ samples via techniques like flipping, zooming, and restoration.
* [ ] **Preprocessing:** Radiometric normalization, scaling, and multispectral transformation.

### Phase 2: Core Analytics

* [ ] **Feature Engineering:** Extraction of GLCM texture, spectral statistics, and morphology.
* [ ] **Spectral Indices:** Implementation of NDVI, EVI, and SAVI.
* [ ] **Modeling:** Comparative analysis between Classical ML (SVM/RF) and Deep Learning (U-Net).

### Phase 3: Temporal & Event Analysis

* [ ] **Event Comparison:** Focused analysis on the March–April 2022 Hasdeo events.
* [ ] **Time-Lapse Visualization:** Frame-by-frame change detection over a 5-year timeline.
* [ ] **Urban Metrics:** Feature visualization over time to assist in city planning.


## Quick Start

### Prerequisites

* Python 3.9+
* Spatial data libraries (e.g., Rasterio, GDAL)
* PyTorch / TensorFlow

### Installation

```bash
git clone https://github.com/yourusername/Urban-Leaf-Health-Monitoring.git
cd Urban-Leaf-Health-Monitoring
pip install -r requirements.txt

```


## Evaluation Metrics

We evaluate our models based on:

* **Segmentation:** Mean Intersection over Union (mIoU) and Dice Coefficient.
* **Classification:** Precision-Recall curves and Confusion Matrices.
* **Temporal:** Quantitative land-cover loss over time.


## Contributing

Contributions are welcome! If you'd like to help with **Region #2** selection or improve our augmentation pipeline, please open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Manjushwarofficial/Urban-Leaf-Health-Monitoring/blob/main/LICENSE) file for details.

