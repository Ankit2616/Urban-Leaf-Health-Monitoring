# Urban Leaf Monitoring Model Pipeline Guide

## 1. Document Purpose

This document explains the **non-web technical side** of the Urban Leaf Monitoring project. It focuses on the machine-learning and remote-sensing pipeline behind the website.

The goal of this document is to clearly explain:

- what model pipeline was built,
- how the project data flows from collection to inference,
- what each main code file does,
- what technical choices were made,
- what work is complete,
- what is still pending for a production-grade scientific system.

---

## 2. Technical Project Summary

The model side of this project is an end-to-end remote-sensing workflow for vegetation and land-cover monitoring using multi-band satellite imagery.

The backend pipeline is designed to:

1. collect Sentinel-2 multi-band scenes,
2. apply cloud filtering and spectral feature generation,
3. convert the scenes into normalized training patches,
4. expand the dataset using augmentation,
5. train an Attention U-Net segmentation model,
6. run inference to generate masks, overlays, and class summaries.

This pipeline supports the scientific and analytical logic later exposed in the website.

---

## 3. Main Problem Solved by the Backend

The non-web side solves this problem:

**How can we process large-scale multi-spectral remote-sensing data into a machine-learning-ready pipeline for vegetation and disturbance segmentation?**

This is important because raw satellite scenes are not directly usable for model training. They must be:

- filtered,
- normalized,
- spatially divided into patches,
- augmented,
- transformed into consistent tensors,
- fed into a model that can perform dense pixel-level prediction.

---

## 4. End-to-End Pipeline Overview

The backend pipeline follows these phases:

### Phase 1: Data Collection

Download Sentinel-2 multi-band TIFF data for Hasdeo from Google Earth Engine.

### Phase 2: Preprocessing

Read TIFF scenes, validate pixels, extract patches, and normalize data.

### Phase 3: Augmentation

Expand the training set with spatial and spectral transformations.

### Phase 4: Model Architecture

Define the Attention U-Net segmentation model.

### Phase 5: Training

Train the segmentation model using AMP, validation, checkpointing, and metric tracking.

### Phase 6: Inference and Visualization

Run the trained model on `.npy` or GeoTIFF inputs and export masks, overlays, and summaries.

---

## 5. Backend File Structure

The core model pipeline lives inside `h100_config/`.

```text
h100_config/
├── 01_data_collection.py
├── 02_preprocessing.py
├── 03_augmentation.py
├── 04_model.py
├── 05_train.py
├── 06_predict_visualize.py
├── setup_env.sh
├── HPC_SETUP_GUIDE.md
├── job_cpu.pbs
├── job_gpu.pbs
└── LOCAL_TEST.py
```

### Meaning of each file

- `01_data_collection.py`  
  Downloads raw multi-band satellite TIFF scenes.

- `02_preprocessing.py`  
  Converts raw scenes into normalized patches.

- `03_augmentation.py`  
  Generates extra training samples for robustness.

- `04_model.py`  
  Defines the Attention U-Net segmentation architecture and loss functions.

- `05_train.py`  
  Runs model training and validation.

- `06_predict_visualize.py`  
  Runs inference and exports visualization artifacts.

- `setup_env.sh`, `job_cpu.pbs`, `job_gpu.pbs`  
  Support HPC environment setup and batch execution.

---

## 6. Phase 1: Data Collection

### Main file

- `h100_config/01_data_collection.py`

### Main purpose

This file collects high-quality Sentinel-2 satellite imagery from Google Earth Engine for the Hasdeo study area.

### Main technical steps

1. initialize Google Earth Engine,
2. define the Hasdeo geographic bounding box,
3. filter the Sentinel-2 image collection by date and cloud percentage,
4. apply cloud and cirrus masking using `QA60`,
5. compute additional spectral indices,
6. download multi-band GeoTIFF outputs.

### Bands collected

The project collects 10 spectral bands:

- B2
- B3
- B4
- B5
- B6
- B7
- B8
- B8A
- B11
- B12

It also computes and appends 4 indices:

- NDVI
- EVI
- SAVI
- BSI

So the full model input later becomes **14 channels**.

### Important code decisions

#### Cloud threshold

The file uses a strict cloud filter:

- `CLOUD_PERCENT = 10`

This improves data reliability for vegetation analysis.

#### Time range

The collection range is broad:

- `2018-01-01` to `2024-12-31`

This supports temporal analysis rather than only one snapshot.

#### Download strategy

Instead of downloading separate files per band, the script uses:

- `filePerBand = False`

This creates a single multi-band TIFF per scene, which is much easier to process consistently later.

### Why this phase matters

This phase builds the raw scientific base of the project. Without good multi-band imagery, later segmentation and feature engineering would be weak or unreliable.

---

## 7. Phase 2: Preprocessing

### Main file

- `h100_config/02_preprocessing.py`

### Main purpose

This file converts raw TIFF scenes into machine-learning-ready patches.

### Main technical steps

1. read raw TIFF images with `rasterio`,
2. build valid-pixel masks,
3. clip values to physical ranges,
4. extract overlapping patches,
5. fill invalid pixels using band medians,
6. normalize each patch band-wise,
7. save the result as `.npy`.

### Key configuration choices

- patch size: `256 x 256`
- stride: `128`
- minimum valid ratio: `0.85`
- total channels used: `14`

### Why patch extraction is important

Remote-sensing scenes are very large. A model cannot directly train on a full landscape image efficiently. Patch extraction:

- standardizes input size,
- increases dataset size,
- improves GPU training efficiency,
- preserves local land-cover structure.

### Why overlapping stride is used

Using a stride smaller than the patch size creates overlap. This helps:

- cover boundaries better,
- avoid losing useful edge information,
- improve training diversity.

### Why normalization matters

Different spectral bands have different value ranges. Z-score normalization keeps the input stable and makes model optimization easier.

### Main important code concepts

#### `BAND_STATS`

Per-band mean and standard deviation values are used to normalize data.

#### `compute_stats()`

This utility computes per-band statistics from a representative sample of TIFF files.

#### `extract_patches()`

This is one of the core functions in the preprocessing pipeline. It:

- scans the scene spatially,
- checks valid-pixel ratio,
- fills masked values,
- returns model-ready patch arrays.

### Why this phase matters

Preprocessing is one of the most important parts of the entire pipeline. A good model can still fail if the input preparation is poor.

---

## 8. Phase 3: Augmentation

### Main file

- `h100_config/03_augmentation.py`

### Main purpose

This file increases training diversity and improves generalization by creating multiple transformed versions of each patch.

### Main augmentation ideas

#### Spatial augmentations

- horizontal flip
- vertical flip
- 90-degree rotation
- transpose
- shift, scale, rotate
- random crop

#### Pixel-level augmentations

- Gaussian noise
- Gaussian blur
- brightness and contrast shifts

#### Spectral augmentations

- `SpectralJitter`
- `RandomBandDrop`

### Why spectral augmentation is important here

This is not a normal RGB-only project. Remote-sensing data benefits from spectral robustness. Spectral augmentation simulates:

- atmospheric differences,
- sensor noise,
- missing band behavior,
- mild acquisition variability.

### Important custom classes

#### `SpectralJitter`

Applies small per-band multiplicative variation to spectral channels. This makes the model less brittle to small sensor or environment shifts.

#### `RandomBandDrop`

Randomly zeros out a small number of non-critical bands. This helps the model tolerate partial signal dropout.

### Dataset expansion target

The script targets:

- around 4 to 5 versions per raw patch
- around 4000 to 5000 total augmented samples

### Why this phase matters

Augmentation improves model robustness, especially when manually labeled remote-sensing data is limited.

---

## 9. Phase 4: Model Architecture

### Main file

- `h100_config/04_model.py`

### Main purpose

This file defines the segmentation architecture used in the project.

### Model used

**Attention U-Net**

### Input shape

- `(B, 14, 256, 256)`

### Output shape

- `(B, 5, 256, 256)`

### Output classes

1. Vegetation
2. Buildings
3. Soil
4. Water
5. Urban Expansion

Note: some surrounding scripts use a slightly different class wording such as `Sparse Vegetation`, `Bare Soil/Rock`, `Built-up/Urban`, and `Water/Shadow`. The important point is that the system is built around **5 semantic land-cover classes**.

### Main building blocks in the code

#### `DoubleConv`

Two sequential convolution blocks with:

- convolution,
- batch normalization,
- ReLU activation.

This is the standard feature extraction building block.

#### `Down`

Performs:

- max pooling,
- then feature extraction through `DoubleConv`.

This reduces spatial size and increases semantic abstraction.

#### `Up`

Performs:

- upsampling,
- spatial alignment,
- skip-connection concatenation,
- then `DoubleConv`.

This restores spatial detail while using encoder information.

#### `AttentionGate`

This is the key enhancement over plain U-Net. It helps the decoder focus on relevant regions instead of treating all skip information equally.

This is useful in dense forest and mixed land-cover scenes where not all texture is equally important.

#### `UNet`

This is the full model class that wires the encoder, attention gates, decoder, and output head together.

### Why Attention U-Net is a strong choice here

- good for pixel-wise segmentation,
- preserves boundary details,
- works well with limited training data,
- attention improves selective focus,
- practical for multispectral remote-sensing scenes.

### Parameter inspection

The model includes:

- `count_parameters()`

This is helpful for reporting model complexity.

---

## 10. Loss Design

### Main classes

- `DiceLoss`
- `CombinedLoss`

### Why not use only Cross Entropy?

Cross Entropy is useful, but segmentation often suffers from class imbalance. For example:

- vegetation may dominate many patches,
- built-up or water may be relatively small,
- disturbance boundaries may be thin and difficult.

### Why Dice Loss helps

Dice Loss rewards overlap quality and helps when minority classes matter.

### Why Combined Loss is good here

The model uses:

- 50% Cross Entropy
- 50% Dice Loss

This balances:

- per-pixel classification behavior,
- region-overlap quality.

---

## 11. Phase 5: Training Pipeline

### Main file

- `h100_config/05_train.py`

### Main purpose

This file trains the model using the augmented patch dataset.

### Main engineering features

- dynamic module loading for `04_model.py`,
- dataset and pseudo-label generation,
- validation split,
- AMP training,
- gradient accumulation,
- gradient clipping,
- cosine annealing learning-rate schedule,
- checkpoint saving,
- resume support,
- early stopping,
- training history export.

### Important code sections

#### `load_model_module()`

Loads `04_model.py` even though the filename starts with a number. This is a practical engineering solution.

#### `VegetationDataset`

This class loads `.npy` patches and creates labels.

### Important honesty point

The current training pipeline uses **pseudo-label generation** instead of expert ground-truth masks.

That means:

- the code pipeline is real and complete,
- the model training loop is real,
- but final scientific benchmarking still depends on replacing pseudo-labels with proper annotation masks.

This is one of the most important points to explain honestly in a panel.

### Pseudo-label logic

The dataset generates classes using thresholds over:

- NDVI
- BSI
- blue-band behavior

This gives a workable bootstrapping label strategy when full manual masks are not yet available.

### Why pseudo-labeling was useful

- it allowed the training pipeline to be completed,
- it enabled end-to-end experimentation,
- it created a prototype training system without waiting for a fully annotated dataset.

### Training settings

Some key defaults:

- epochs: `100`
- batch size: `16`
- gradient accumulation: `2`
- learning rate: `3e-4`
- weight decay: `1e-4`
- validation split: `0.15`

### H100 optimization choices

The script enables:

- TF32 matmul,
- TF32 cuDNN,
- cuDNN benchmark,
- AMP with GradScaler.

These are strong practical choices for GPU training efficiency.

### Important functions in the training file

#### `train_one_epoch()`

Handles:

- forward pass,
- mixed precision,
- scaled backpropagation,
- accumulation logic,
- gradient clipping,
- optimizer stepping.

#### `validate()`

Computes:

- validation loss,
- mean IoU,
- pixel accuracy.

#### `class_distribution()`

Creates a class-ratio report for pseudo-label distribution. This is useful for understanding imbalance.

### Outputs generated by training

- `best_model.pth`
- periodic checkpoints
- `training_history.json`
- `training_history.csv`
- `pseudo_label_distribution.json`

### Why this phase matters

This phase is where the project moves from data preparation to actual model learning.

---

## 12. Metrics Used

### Mean Intersection over Union (mIoU)

This is the main segmentation quality metric. It measures overlap between predicted and true regions.

### Pixel Accuracy

Measures how many pixels were classified correctly overall.

### Dice-based optimization

Although Dice is used as a loss component, it is also conceptually important for overlap-oriented segmentation quality.

### Why multiple metrics matter

One metric alone can hide weaknesses. For example:

- pixel accuracy can look high if one class dominates,
- IoU gives a better sense of region quality,
- Dice helps optimize overlap robustness.

---

## 13. Phase 6: Inference and Visualization

### Main file

- `h100_config/06_predict_visualize.py`

### Main purpose

This file runs the trained model on new input scenes and exports outputs that can be used in reports, GIS workflows, or the website.

### Supported inputs

- `.npy`
- `.tif`
- `.tiff`

### Main technical behavior

1. load trained checkpoint,
2. normalize input,
3. tile the image into windows,
4. run batch prediction,
5. aggregate votes across overlapping windows,
6. generate final mask,
7. export mask, overlay, summary, and TIFF output.

### Important code components

#### `predict_array()`

This is one of the central functions of the inference pipeline. It:

- pads the input if required,
- creates overlapping windows,
- runs predictions in batches,
- merges local predictions using vote accumulation.

This is very important for large scenes because it avoids forcing the model to process an oversized image in one pass.

#### `summarize_mask()`

Returns per-class pixel counts and ratios. These are useful for reports and dashboard summaries.

#### `save_outputs()`

Exports:

- `.npy` mask
- `.png` mask
- `.json` summary
- optional `.tif` mask
- `.png` overlay

### Why this phase matters

This phase makes the model outputs usable outside the training loop. It is the bridge from model weights to practical interpretation.

---

## 14. Code Design Strengths

The backend code has several strong engineering ideas.

### 14.1 Clear phase separation

Each major pipeline stage has its own file. This makes debugging and explanation much easier.

### 14.2 Reproducibility

- statistics are saved,
- metadata is saved,
- training history is saved,
- checkpoints are saved.

### 14.3 HPC readiness

The project includes environment setup and PBS job scripts, showing that it was designed for large-scale training rather than only local experimentation.

### 14.4 Model modularity

The architecture file is separate from the training file, which is a good design choice for experimentation.

### 14.5 Artifact generation

The inference stage generates all the artifact types needed for later integration:

- masks,
- overlays,
- summaries,
- checkpoint references.

That is why the web layer can later surface model results cleanly.

---

## 15. Work Completed on the Model Side

### Completed work

- Hasdeo data collection pipeline designed
- cloud masking and spectral band selection implemented
- vegetation and soil indices added
- preprocessing and patch extraction implemented
- normalization pipeline implemented
- augmentation pipeline implemented
- Attention U-Net architecture implemented
- combined loss implemented
- training loop implemented
- validation and metrics implemented
- checkpointing and resume support implemented
- inference and visualization pipeline implemented
- HPC execution support added

### Work still pending for final scientific maturity

- replace pseudo-labels with manually verified ground-truth masks
- run final benchmark experiments on held-out labels
- record stable final metrics for publication-quality reporting
- connect real checkpoint artifacts directly into the live segmentation web page

---

## 16. Important Main Points in the Code

These are the most important code-level ideas a panel may ask about.

### Point 1: The model is multispectral, not RGB-only

This is one of the strongest points in the codebase. The pipeline is designed around 14 channels, which is much more meaningful for remote-sensing segmentation.

### Point 2: Attention was added to improve focus

The architecture is not a plain CNN. Attention gates are used to improve selective feature use in complex scenes.

### Point 3: Patch-based training is deliberate

Large scenes are broken into patches because remote-sensing images are too large to use directly and because patching increases training efficiency.

### Point 4: Augmentation is domain-aware

The augmentation is not just random image flipping. It includes spectral jitter and band dropout, which are much more relevant to multispectral imagery.

### Point 5: Pseudo-labeling is a practical bootstrap strategy

The system can already train end to end, even before a full manual annotation pipeline is ready.

### Point 6: Inference exports are dashboard-ready

The output format is intentionally practical for visualization, reporting, and future web integration.

---

## 17. How This Model Side Supports the Website

The website uses the model-side work in two ways.

### 17.1 Directly

The model-result API is already prepared to detect:

- trained checkpoints,
- training history,
- mask outputs,
- overlay outputs,
- JSON summaries.

### 17.2 Conceptually

Even where the website currently uses curated or heuristic views, its logic is based on the same remote-sensing pipeline:

- land-cover classes,
- vegetation indicators,
- disturbance interpretation,
- temporal comparison,
- segmentation explanation.

---

## 18. Limitations and Honest Positioning

This project should be presented honestly as:

**a strong end-to-end prototype and engineering pipeline with a near-complete scientific workflow**

It should not be presented as:

**a fully benchmarked final ecological segmentation product**

The reason is simple:

- architecture is real,
- training pipeline is real,
- inference pipeline is real,
- but final supervised labels and benchmark metrics are still the scientific gap.

This is not a weakness in code quality. It is a data-validation gap.

---

## 19. Likely Panel Questions and Good Answers

### Q1. What is the model used in this project?

**Answer:**  
The project uses an Attention U-Net designed for 14-channel remote-sensing segmentation with 5 output classes.

### Q2. Why not use a normal CNN classifier?

**Answer:**  
Because classification only predicts one label for the whole image, while this project needs pixel-level understanding of land-cover structure. That requires segmentation.

### Q3. Why did you use 14 bands?

**Answer:**  
Vegetation, soil, water, and built-up regions are better separated when multispectral information and derived indices are used instead of only RGB.

### Q4. What is the role of preprocessing?

**Answer:**  
Preprocessing standardizes the data, removes poor-quality pixels, extracts consistent patches, and prepares tensors that the model can learn from reliably.

### Q5. Why is augmentation important here?

**Answer:**  
Remote-sensing data is limited and variable. Augmentation improves generalization and helps the model tolerate spatial and spectral variation.

### Q6. What makes your augmentation domain-aware?

**Answer:**  
The pipeline includes spectral jitter and band drop, which are more meaningful for multispectral satellite data than plain RGB-only image augmentation.

### Q7. What loss function did you use?

**Answer:**  
The project uses a combined loss: Cross Entropy plus Dice Loss. This improves segmentation quality, especially under class imbalance.

### Q8. What metrics do you track?

**Answer:**  
The main metrics are validation loss, mean IoU, and pixel accuracy. Dice is also important as part of the optimization strategy.

### Q9. What is pseudo-labeling in your pipeline?

**Answer:**  
Pseudo-labeling means the current masks are generated from spectral thresholds instead of expert manual annotation. It allows prototype training while the full labeling pipeline is still pending.

### Q10. Is pseudo-labeling a limitation?

**Answer:**  
Yes, it is the main scientific limitation. The engineering pipeline is complete, but final benchmark confidence requires real ground-truth masks.

### Q11. Why did you use Attention U-Net?

**Answer:**  
It preserves spatial detail through skip connections and improves focus on relevant regions using attention gates, which is helpful in mixed forest and disturbed landscapes.

### Q12. How does inference work on large scenes?

**Answer:**  
Large scenes are processed as overlapping windows, predicted in batches, and then merged using vote aggregation to form a final segmentation mask.

### Q13. What outputs are generated after inference?

**Answer:**  
The inference script generates masks, overlays, class-ratio summaries, NumPy outputs, and optional GeoTIFF masks.

### Q14. What part of the model code is strongest?

**Answer:**  
The strongest parts are the multispectral design, attention-based segmentation, domain-aware augmentation, and the practical inference artifact pipeline.

### Q15. What is the biggest next step?

**Answer:**  
The biggest next step is replacing pseudo-labels with expert-annotated masks and then reporting final supervised benchmark metrics.

---

## 20. Final Conclusion

The model side of Urban Leaf Monitoring is a serious technical pipeline that covers:

- data collection,
- preprocessing,
- augmentation,
- architecture design,
- training,
- validation,
- inference,
- and export-ready output generation.

Its biggest strength is that it is designed for **multispectral remote-sensing segmentation**, not just ordinary RGB experimentation.

Its biggest remaining gap is **ground-truth validation quality**, not engineering completeness.

That is an important and defensible position in a project presentation: the pipeline is real, the architecture is meaningful, the outputs are usable, and the scientific next step is clearly identified.
