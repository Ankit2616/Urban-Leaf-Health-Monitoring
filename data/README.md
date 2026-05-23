# Data Extraction and Processing

This folder contains all the data and resources required for the Urban Leaf Health Monitoring project. The data is organized and processed to achieve the project's objectives, including comparison, segmentation, and classification.

## Objectives
- [x] Comparison based on events (2018–2023: adds 40k trees cutting) by visualizing features using GEE APIs.
- [ ] Comparison over years (e.g., showing in frames).
- [x] Urban city planning based on convenience (optional).
- [x] Base paper (2024/25) SCI Journal, Google Scholar.

## Data Processing Workflow

### 1. Region: (ML Model Being Used)
- [x] Gather data for Objective 1.
- [x] Verification/ground truthing.
- [x] Data collection (at least 1000 raw images after cloud masking, .tiff format extension, high quality).
- [ ] Convert data into JPEG format and create a catalog.
- [ ] Apply image augmentation techniques (e.g., zooming, flipping, restoration) to generate 4000–5000 images or more based on the number of raw images.
- [ ] Perform preprocessing tasks such as scaling and transformation.
- [ ] Conduct feature engineering and feature selection.
- [ ] Perform comparison, segmentation, and classification (method undecided).

## Folder Structure
- **01_area_of_interest_data_verification/**: Contains initial and verified images of the area of interest.
- **01_area_of_interest_selection_using_sampling/**: Includes sampled data batches for different regions.
- **02_comparison_based_on_events/**: Contains event-based comparison data, including cleaned image features and metadata.
- **03_hasdeo_1000_all_bands/**: Logs and metadata for the Hasdeo 1000 dataset.

## Steps for Data Processing
1. **Data Collection**: Use GEE APIs to collect raw images in .tiff format.
2. **Cloud Masking**: Apply cloud masking to ensure high-quality images.
3. **Conversion**: Convert .tiff images to JPEG format.
4. **Augmentation**: Apply techniques like zooming, flipping, and restoration to expand the dataset.
5. **Preprocessing**: Scale and transform the data for analysis.
6. **Feature Engineering**: Select and engineer features for the ML model.
7. **Analysis**: Perform segmentation, classification, and comparison tasks.

---

For any queries or contributions, feel free to raise an issue or submit a pull request.