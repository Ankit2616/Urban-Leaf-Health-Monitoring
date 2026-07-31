# Urban Leaf Monitoring Website Guide

## 1. Project Title

**Urban Leaf Monitoring: Remote-Sensing Dashboard for Deforestation Analysis, Segmentation Demonstration, and Lower-Impact Development Recommendation**

---

## 2. Project Summary

This project is a web-based decision-support system built around remote-sensing evidence from the Hasdeo forest region and related curated scenes. The website does not act as a generic "upload one image and get one prediction" demo. Its main purpose is to:

1. monitor vegetation and land-cover changes across time,
2. visually demonstrate how segmentation can interpret a satellite or RGB scene,
3. explain the project hypothesis with evidence,
4. recommend lower-impact planning zones near Hasdeo using a transparent scoring method.

The application combines:

- curated satellite imagery,
- monthly or event-based feature tables,
- a segmentation model pipeline designed for multi-band remote-sensing input,
- a website layer that makes the outputs understandable to a non-technical reviewer.

---

## 3. Core Problem Statement

The main problem addressed by this project is:

**How can remote sensing, spectral indicators, and segmentation-based scene understanding help identify ecological disturbance and support lower-impact planning decisions near forested landscapes such as Hasdeo?**

This is important because direct field inspection is expensive, slow, and difficult to repeat at large scale. Satellite-based monitoring allows repeatable observation of:

- canopy degradation,
- vegetation health decline,
- exposed soil increase,
- built-up pressure,
- thermal stress,
- recovery behavior over time.

---

## 4. Main Goal of the Website

The website has four connected goals.

### 4.1 Timeline monitoring

Show how a selected region changes across a selected time window using vegetation, soil, water, urban, and thermal indicators.

### 4.2 Segmentation explanation

Show how scene segmentation works visually so that users can inspect masks, overlays, class ratios, and derived indicators such as canopy share and built-up share.

### 4.3 Hypothesis explanation

Explain the Part 1 research hypothesis in a simple but evidence-backed way.

### 4.4 Recommendation support

Move beyond monitoring and answer a planning question:

**Which side or zone should be treated as more suitable, conditional, or avoidable if ecological disruption is to be minimized?**

---

## 5. Major Website Pages

### 5.1 Timeline Page (`/`)

This is the main dashboard page. It is the operational center of the web project.

#### What this page does

- lets the user choose a region,
- lets the user choose a feature such as NDVI or canopy cover,
- lets the user select a before/after time window,
- compares the selected periods,
- shows nearest curated images,
- plots timeline charts,
- shows signal cards for vegetation, canopy, heat, built-up pressure, soil exposure, and moisture.

#### Why it matters

This page converts raw feature tables into a readable change-analysis interface. It helps the user inspect whether a region is recovering, degrading, or showing warning signals.

#### Key user value

The page makes multi-indicator interpretation fast. Instead of reading a CSV manually, the user sees:

- numerical change,
- supporting visual evidence,
- grouped charts,
- a short interpretation summary.

---

### 5.2 Segmentation Lab (`/segmentation-lab`)

This page is built to explain segmentation clearly.

#### What this page does

- accepts an uploaded image or uses curated sample scenes,
- runs two segmentation flows on the same frame,
- shows input, overlay, and mask,
- compares class distributions,
- derives canopy cover, exposed surface, built-up share, and water/shadow share,
- reports agreement between both outputs.

#### Why this page exists

A project panel often asks:

> "How is your model understanding the image?"

The Segmentation Lab answers that visually.

#### Important honesty note

At present, the website demo uses two lightweight RGB-based segmentation workflows inside the browser:

- a **reference segmenter** used as an external-style baseline,
- an **Urban Leaf workflow** that uses project-oriented scene rules and indicator extraction.

This allows immediate interaction on the website.

The **production segmentation backend** intended for the real project is the 14-channel Attention U-Net defined in `h100_config/04_model.py`, trained through `h100_config/05_train.py`, and run for inference through `h100_config/06_predict_visualize.py`.

So the current web segmentation page is:

- a **functional visual comparison interface today**,
- and a **ready UI contract for plugging in the trained model backend**.

---

### 5.3 Recommendation Page (`/recommendation`)

This page answers the planning question more directly.

#### What this page does

- divides the Hasdeo study region into named zones,
- ranks zones by suitability,
- labels them as `Preferred zone`, `Conditional`, or `Avoid`,
- shows a clickable zone map,
- explains why a zone received its score.

#### Main scoring ideas used

- overlap with forest core,
- centrality inside the main forest block,
- periphery advantage,
- evidence confidence from export coverage.

#### Why this page matters

This is the layer that makes the project useful for real discussion. It moves the system from:

**"What changed?"**

to:

**"Where should activity be avoided, and where is pressure relatively less damaging?"**

#### Important honesty note

This recommendation engine is currently a **transparent heuristic decision-support layer**, not a fully GIS-validated environmental clearance model. It is still useful because it:

- uses real project region structure,
- uses real Hasdeo bounds,
- uses real feature trends,
- shows reasoning openly instead of pretending to be exact beyond the available data.

---

### 5.4 Hypothesis Page (`/hypothesis`)

This page explains the scientific framing behind the entire project.

#### Core hypothesis

**Forest degradation can be detected through synchronized drops in vegetation indices and rises in exposed-land signals.**

#### What the page explains

- study area choice,
- observation stack,
- expected signal behavior,
- event focus,
- decision rule for degradation interpretation.

#### Why this page matters

This page is important during presentation because it explains that the website is not just a UI. It is a visual interface for a research argument.

---

## 6. Backend Model Used for Segmentation

The main segmentation architecture prepared for this project is in:

- `h100_config/04_model.py`

### Model name

**Attention U-Net**

### Input design

- input channels: **14**
- image size: **256 x 256**
- classes: **5**

### Classes used

1. Vegetation
2. Sparse Vegetation
3. Bare Soil / Rock
4. Built-up / Urban
5. Water / Shadow

### Why this model was chosen

Attention U-Net is suitable because:

- U-Net is strong for pixel-level segmentation,
- skip connections preserve spatial detail,
- attention gates help focus on relevant areas,
- it works well for land-cover segmentation where context and boundaries matter,
- it is a practical architecture for remote-sensing scenes with many input bands.

### Why 14 channels were used

The design is not limited to RGB only. It is intended to use:

- optical bands,
- near-infrared bands,
- short-wave infrared bands,
- vegetation and soil-related indices.

That is much more suitable for remote sensing than a plain RGB-only model.

---

## 7. Training and Inference Pipeline

### 7.1 Training script

- `h100_config/05_train.py`

### Main training features

- AMP / mixed precision,
- gradient accumulation,
- checkpointing,
- validation tracking,
- early stopping,
- pseudo-label support for current dataset flow.

### Loss design

The model uses **Combined Loss**:

- Cross Entropy Loss
- Dice Loss

This is useful because:

- cross entropy supports class prediction accuracy,
- dice loss helps with class imbalance,
- land-cover segmentation often has uneven class distribution.

### 7.2 Inference script

- `h100_config/06_predict_visualize.py`

### What inference generates

- mask outputs,
- overlay outputs,
- summary JSON files,
- optional TIFF masks for GIS work.

### Why these outputs matter to the website

The website can automatically surface:

- trained checkpoint status,
- segmentation masks,
- overlay previews,
- per-class ratios,
- run summaries.

This happens through the API route:

- `app/api/model-results/route.ts`

---

## 8. Why There Are Two Segmentation Outputs on the Website

The Segmentation Lab shows:

1. **Reference segmenter**
2. **Urban Leaf workflow**

### Why compare two outputs?

Because comparison is useful in a presentation and in development:

- it gives a visual baseline,
- it shows that our project logic is not blindly trusted,
- it helps explain what changes when a project-specific interpretation is used,
- it prepares the website for future "our model vs external baseline" evaluation.

### Why not only one segmentation output?

Only one output does not help much in explanation. Two outputs make it easier to discuss:

- similarity,
- disagreement,
- class differences,
- strengths of a project-oriented pipeline.

### Important practical note

At the current website stage, the comparison is designed for clear visual explanation and interaction speed. The production upgrade path is to replace the project-side heuristic with actual checkpoint inference from the Attention U-Net backend.

---

## 9. Indicators Used Across the Website

The dashboard uses the following feature indicators:

- NDVI
- NDWI
- MNDWI
- NDBI
- EVI
- SAVI
- LAI
- GCI
- ARVI
- BSI
- LST (Celsius)
- canopy cover percentage

### Why these indicators matter

#### Vegetation indicators

- NDVI
- EVI
- SAVI
- LAI
- GCI
- ARVI

These help detect vegetation density, chlorophyll condition, and canopy health.

#### Water indicators

- NDWI
- MNDWI

These help interpret moisture and water-related scene behavior.

#### Urban or exposed-surface indicators

- NDBI
- BSI

These help detect built-up pressure, bare ground, and disturbance.

#### Thermal indicator

- LST

This helps detect temperature rise associated with stress and exposed surface conditions.

---

## 10. Recommendation Logic for Hasdeo

The recommendation logic is implemented in:

- `lib/recommendation.ts`

### Study zones used

- Hasdeo Buffer
- Hasdeo West
- Hasdeo East
- Hasdeo North
- Hasdeo South
- Hasdeo Core

### Factors used in scoring

#### 10.1 Core overlap

If a zone overlaps the ecological core, it is penalized heavily.

#### 10.2 Centrality

Zones near the center of the forest block are treated as riskier because interior fragmentation is more harmful.

#### 10.3 Periphery advantage

Zones closer to the outer edge are favored because they are less likely to cut through the forest interior.

#### 10.4 Evidence confidence

Export coverage is used as a confidence signal so that the recommendation has some support from real project data coverage.

### Current output meaning

- **Preferred zone**: lower relative ecological disruption under the current heuristic
- **Conditional**: use only with caution and more detailed assessment
- **Avoid**: high ecological sensitivity or strong structural penalty

---

## 11. Website Architecture

### Frontend

- Next.js 14
- React 18
- TypeScript
- custom CSS
- Lucide icons

### Data and app logic

- server-side data loading in `lib/data.ts`
- recommendation engine in `lib/recommendation.ts`
- shared types in `lib/shared.ts`

### API layer

- `app/api/regions/route.ts`
- `app/api/features/route.ts`
- `app/api/compare/route.ts`
- `app/api/images/route.ts`
- `app/api/asset/route.ts`
- `app/api/model-results/route.ts`
- `app/api/recommendation/route.ts`

### Why this architecture is good

- clean page separation,
- reusable data layer,
- API endpoints can later connect to a proper backend or database,
- UI remains independent from heavy training code,
- segmentation model artifacts can be surfaced without changing the UI structure.

---

## 12. Current Data Source Strategy

The website currently works from curated project data already inside the repository.

### That includes

- cleaned feature CSV files,
- curated JPEG scenes,
- sample RGB scenes,
- metadata manifests,
- export logs.

### Why this is useful

It gives a stable presentation environment. Users do not depend on live satellite APIs during the demo.

---

## 13. File Structure

Below is the important structure for the website and model pipeline.

```text
Urban Leaf Monitoring/
├── app/
│   ├── layout.tsx
│   ├── page.tsx
│   ├── globals.css
│   ├── hypothesis/page.tsx
│   ├── segmentation-lab/page.tsx
│   ├── recommendation/page.tsx
│   └── api/
│       ├── asset/route.ts
│       ├── compare/route.ts
│       ├── features/route.ts
│       ├── images/route.ts
│       ├── model-results/route.ts
│       ├── recommendation/route.ts
│       └── regions/route.ts
├── components/
│   ├── site-header.tsx
│   ├── timeline-dashboard.tsx
│   └── recommendation-dashboard.tsx
├── lib/
│   ├── data.ts
│   ├── recommendation.ts
│   └── shared.ts
├── h100_config/
│   ├── 01_data_collection.py
│   ├── 02_preprocessing.py
│   ├── 03_augmentation.py
│   ├── 04_model.py
│   ├── 05_train.py
│   ├── 06_predict_visualize.py
│   └── HPC_SETUP_GUIDE.md
├── data/
│   ├── 01_area_of_interest_data_verification/
│   ├── 01_area_of_interest_selection_using_sampling/
│   ├── 02_comparison_based_on_events/
│   └── 03_hasdeo_1000_all_bands/
└── docs/
    └── Urban_Leaf_Web_Project_Guide.md
```

---

## 14. Tech Stack

### Website stack

- Next.js 14
- React 18
- TypeScript
- CSS
- Lucide React

### Model and analysis stack

- Python
- PyTorch
- NumPy
- Matplotlib
- optional Rasterio for GeoTIFF inference

### Why this stack was chosen

- Next.js gives fast page routing and API integration,
- TypeScript keeps UI logic safer,
- PyTorch is suitable for research and segmentation work,
- Python scripts are more natural for preprocessing, training, and inference,
- the stack is practical for a student project that still aims to look industry-grade.

---

## 15. What Is Fully Functional Today

### Functional now

- timeline monitoring page,
- feature comparison APIs,
- curated image serving,
- segmentation comparison page,
- hypothesis page,
- Hasdeo recommendation page,
- model artifact detection route,
- recommendation API.

### Functional but currently heuristic or staged

- recommendation engine is heuristic rather than a final ecological model,
- segmentation web demo is browser-based rather than real model inference,
- model-result lane depends on actual checkpoint and inference artifacts being present in the repo.

---

## 16. What Can Be Upgraded Next

1. connect the real trained checkpoint to the Segmentation Lab,
2. store inference outputs in `results/` and surface them automatically,
3. replace zone heuristics with denser grid-based or polygon-based scoring,
4. add real database-backed storage if required,
5. add GIS-export support for recommendation zones,
6. add user-authenticated project workspaces if the system becomes multi-user.

---

## 17. Why the Hypothesis and Recommendation Pages Are the Glory of the Project

The timeline page shows **what changed**.

The Segmentation Lab shows **how the system interprets a scene**.

The Hypothesis page shows **why the monitoring logic exists**.

The Recommendation page shows **how the project becomes actionable**.

This combination is the strongest part of the website because it creates a complete story:

1. scientific framing,
2. visual evidence,
3. analytical monitoring,
4. planning recommendation.

That is what makes the project more than a normal dashboard.

---

## 18. Likely Minor Project Panel Questions and Strong Answers

### Q1. What is the main aim of this project?

**Answer:**  
The aim is to use remote-sensing data and web-based analysis to monitor deforestation-related change, demonstrate segmentation-based scene interpretation, and support lower-impact planning decisions in the Hasdeo region.

### Q2. Why did you choose Hasdeo?

**Answer:**  
Hasdeo is a meaningful case because it is ecologically sensitive and relevant to land-cover disturbance discussions. It provides a strong study area for comparing vegetation decline, surface exposure, and planning risk.

### Q3. Is this website only a frontend demo?

**Answer:**  
No. The website is supported by project data, API routes, feature comparison logic, model artifact discovery, and a remote-sensing segmentation pipeline prepared in Python. Some parts are already fully connected, while the segmentation production backend is prepared for integration.

### Q4. Which model did you use for segmentation?

**Answer:**  
The main project model is an Attention U-Net with 14-channel input and 5 output classes. It is defined in `h100_config/04_model.py`.

### Q5. Why did you choose U-Net?

**Answer:**  
U-Net is widely used for segmentation because it preserves spatial detail through skip connections. The attention version improves focus on important regions, which is useful for mixed land-cover scenes.

### Q6. Why 14 channels instead of RGB only?

**Answer:**  
Remote-sensing analysis benefits from spectral information beyond RGB. The 14-channel design allows the model to use optical bands and derived indices that are more informative for vegetation, soil, built-up regions, and water/shadow separation.

### Q7. Is the segmentation on the website using your real trained model right now?

**Answer:**  
The current Segmentation Lab is a live visual comparison module using browser-side RGB workflows for speed and explanation. The production model backend is the 14-channel Attention U-Net, which can be plugged into the same UI through the inference pipeline.

### Q8. Why did you show a reference segmenter and your own workflow together?

**Answer:**  
This helps compare outputs and explain model behavior more clearly. It is useful both for evaluation and for presentation, because visual comparison makes strengths and differences easier to discuss.

### Q9. What are the important indicators in your dashboard?

**Answer:**  
The most important ones are NDVI, EVI, SAVI, LAI, GCI, BSI, NDBI, NDWI, and land surface temperature. Together they help interpret vegetation condition, water signal, exposed ground, built-up pressure, and heat stress.

### Q10. What does the hypothesis page represent?

**Answer:**  
It explains the project’s scientific assumption: degradation should appear as a drop in vegetation-related indices and a rise in exposed-land or urban-related signals during disturbance periods.

### Q11. What is the recommendation page doing?

**Answer:**  
It ranks Hasdeo zones based on ecological sensitivity and planning suitability. It is designed to help answer where development pressure should be minimized and which regions deserve stronger caution.

### Q12. Is the recommendation output fully scientifically final?

**Answer:**  
No. It is a transparent decision-support heuristic based on current project evidence. It is useful for planning discussion, but not a replacement for formal ecological clearance or field validation.

### Q13. How are your APIs useful in this project?

**Answer:**  
The APIs make the UI modular. They return region lists, feature tables, comparison results, images, model artifacts, and recommendation data. This makes the website easier to scale and maintain.

### Q14. What is the biggest technical strength of your system?

**Answer:**  
Its strength is the combination of remote-sensing analysis, model-readiness, clear visualization, and a recommendation layer. Many student projects stop at classification or charts; this one also supports explanation and planning.

### Q15. What is the current limitation?

**Answer:**  
The main limitation is that the fully trained segmentation checkpoint is not yet wired directly into the live web inference flow. The recommendation layer is also heuristic and can be improved with denser geospatial scoring.

### Q16. How would you improve the project next?

**Answer:**  
I would connect the real checkpoint to the Segmentation Lab, generate batch inference artifacts for the dashboard automatically, and replace zone-level heuristics with grid-level ecological scoring.

### Q17. Why is this project relevant in industry?

**Answer:**  
Because organizations need interpretable monitoring and planning tools, not only raw models. This project shows how analysis, model outputs, and decision support can be packaged into a usable product interface.

### Q18. What makes this different from a plain GIS report?

**Answer:**  
A GIS report is usually static. This project is interactive, lets users explore time windows and indicators, provides segmentation demonstration, and exposes recommendation logic in a reusable website form.

### Q19. Why did you not build only a machine-learning notebook?

**Answer:**  
Because decision-makers usually need an interface, not only code. The website translates technical outputs into something inspectable and explainable for non-technical stakeholders.

### Q20. What is the single biggest takeaway from this project?

**Answer:**  
Remote-sensing evidence becomes much more useful when it is turned into an interpretable workflow that goes from hypothesis, to monitoring, to scene understanding, to planning support.

---

## 19. Final Conclusion

This website represents the web-facing operational layer of the Urban Leaf Monitoring project. It combines research framing, change analysis, segmentation explanation, and lower-impact planning guidance into one coherent interface.

It is strong not because it claims everything is complete, but because it shows:

- a meaningful environmental problem,
- a credible remote-sensing workflow,
- a real model design for segmentation,
- an interpretable website experience,
- and a clear path from observation to recommendation.
