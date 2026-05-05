# UAE Walkability Assessment Pipeline

Semantic segmentation-based walkability scoring for UAE urban streets using crowdsourced Mapillary imagery and SegFormer-B2.

---

## Overview

This pipeline evaluates street-level walkability by:
1. Downloading street-level images from Mapillary
2. Running semantic segmentation using SegFormer-B2 (pretrained on ADE20K)
3. Computing walkability scores from sidewalk, vegetation, and obstruction pixel ratios
4. Validating scores against OpenStreetMap infrastructure data
5. Generating visualizations including an interactive map and score distribution plots

---

## Requirements

### Python Version
Python 3.10 or higher recommended.

### Install Dependencies
```bash
pip install torch torchvision transformers pillow pandas numpy scipy tqdm requests folium matplotlib scikit-learn osmnx
```

### Hardware
- The segmentation step (02_segment.py) is computationally intensive
- A GPU is recommended but not required — the code automatically detects and uses CUDA if available
- On CPU, expect approximately 2–4 hours for 1,000 images
- On GPU, expect approximately 20–30 minutes

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/israa-a/uae-walkability-assessment.git
cd uae-walkability-assessment
```

### 2. Create the data directory
```bash
mkdir data
mkdir data/images
```

### 3. Configure paths and credentials
Open `config.py` and set the following:

```python
# Replace with your Mapillary API token
ACCESS_TOKEN = "YOUR_MAPILLARY_TOKEN_HERE"

# Paths (defaults use relative paths and should work as-is)
IMAGE_DIR    = "data/images"
METADATA_CSV = "data/metadata.csv"
SCORES_CSV   = "data/scores.csv"
SUMMARY_CSV  = "data/area_summary.csv"
```

To get a Mapillary API token, create a free account at mapillary.com and generate a token from the Developer section.

---

## Running the Full Pipeline

Run the scripts in the following order:

### Step 1 — Download images from Mapillary
```bash
python 01_download.py
```
- Downloads up to 200 images per area across 10 UAE study areas
- Saves images to `data/images/` and metadata to `data/metadata.csv`
- Expected runtime: 30–60 minutes depending on connection speed

### Step 2 — Run segmentation and compute walkability scores
```bash
python 02_segment.py
```
- Loads SegFormer-B2 from Hugging Face (downloads ~110MB on first run)
- Runs semantic segmentation on all images in `data/images/`
- Computes sidewalk, vegetation, and obstruction pixel ratios
- Applies weighted scoring formula and min-max normalization
- Saves per-image scores to `data/scores.csv`
- Expected runtime: 20–30 min (GPU) or 2–4 hours (CPU)

### Step 3 — Aggregate scores to area level
```bash
python 03_aggregate.py
```
- Computes mean and standard deviation of walkability scores per area
- Saves area-level summary to `data/area_summary.csv`
- Expected runtime: seconds

### Step 4 — Fetch OSM data and validate
```bash
python 05_osm_validate.py
```
- Fetches pedestrian infrastructure, vegetation, and barrier data from OpenStreetMap
- Computes OSM composite scores using the same weighted formula
- Evaluates classification agreement between SegFormer and OSM approaches
- Saves OSM scores to `data/osm_scores.csv` and validation report to `data/validation.txt`
- Expected runtime: 10–20 minutes (requires internet)

### Step 5 — Generate visualizations
```bash
python 04_visualize.py
```
- Generates interactive Folium map (`walkability_map.html`)
- Generates boxplot, bar chart, and distribution plots
- Expected runtime: seconds

---

## Running on Your Own Data

To apply the pipeline to different areas:

1. Open `config.py`
2. Replace the `BBOXES` list with your own bounding boxes in `"west,south,east,north"` format:
```python
BBOXES = [
    "55.270,25.190,55.300,25.220",  # Your area name
]
```
3. Update `AREA_LABELS` to map each bounding box string to a readable name
4. Update `AREA_ORDER` for plot ordering
5. Run the pipeline from Step 1

---

## Using Pre-computed Results (Skip Download and Segmentation)

If you already have a `scores.csv` file from a previous run, you can skip Steps 1 and 2 and run directly from Step 3:

```bash
python 03_aggregate.py
python 05_osm_validate.py
python 04_visualize.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `data/images/` | Downloaded Mapillary images |
| `data/metadata.csv` | Image IDs, GPS coordinates, bounding boxes |
| `data/scores.csv` | Per-image walkability scores and pixel ratios |
| `data/area_summary.csv` | Area-level mean and std walkability |
| `data/osm_scores.csv` | OSM-derived metrics and composite scores |
| `data/validation.txt` | Classification agreement report |
| `walkability_map.html` | Interactive map (open in any browser) |
| `plot_boxplot.png` | Score distributions per area |
| `plot_barchart.png` | SegFormer vs OSM feature comparison |
| `plot_distributions.png` | Per-image KDE distributions vs OSM values |

---

## Scoring Formula

```
raw_score = 0.6 × R_sidewalk + 0.4 × R_vegetation − 1.5 × R_obstruction
walkability = min-max normalize(raw_score) → [0, 1]
```

**ADE20K classes used:**

| Feature | Classes |
|---------|---------|
| Sidewalk | 11 (sidewalk/pavement) |
| Vegetation | 4 (tree), 9 (grass), 17 (plant), 72 (palm) |
| Obstruction | 32 (fence) |

**Walkability categories:**
- High: ≥ 0.53
- Medium: 0.47 – 0.53
- Low: < 0.47

---

## Model

- **Model:** SegFormer-B2 finetuned on ADE20K
- **Checkpoint:** `nvidia/segformer-b2-finetuned-ade-512-512` (Hugging Face)
- The model is downloaded automatically from Hugging Face on first run (~110MB)

---

## Notes

- The Mapillary access token is required only for Step 1. Steps 2–5 do not require it.
- All scores are dataset-relative due to min-max normalization and should not be compared across different runs with different image sets.
- OSM data completeness varies by region. Areas with sparse OSM contributor activity may show artificially low scores.
