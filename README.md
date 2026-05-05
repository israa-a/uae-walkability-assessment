# UAE Walkability Assessment Pipeline

Semantic segmentation-based walkability scoring for UAE urban streets.  

---

## Pipeline Overview

```
01_download.py    → Download images from Mapillary
02_segment.py     → Run SegFormer-B2, compute per-image scores
03_aggregate.py   → Aggregate scores to area level
04_visualize.py   → Generate map and plots
05_osm_validate.py → Fetch OSM data and evaluate classification agreement
```

---

## Setup

### Install dependencies
```bash
pip install torch transformers pillow pandas numpy tqdm requests folium matplotlib scipy scikit-learn osmnx
```

### Configure paths and token
Open `config.py` and set:
- `ACCESS_TOKEN` — your Mapillary API token
- `IMAGE_DIR` — where images will be saved
- `METADATA_CSV`, `SCORES_CSV`, `SUMMARY_CSV` — output paths

---

## Usage

Run each script in order:

```bash
python 01_download.py       # ~30–60 min depending on connection
python 02_segment.py        # ~2–4 hours on CPU, ~20 min on GPU
python 03_aggregate.py      # seconds
python 05_osm_validate.py   # ~10–20 min (requires internet)
python 04_visualize.py      # seconds
```

---

## Outputs

| File | Description |
|------|-------------|
| `data/metadata.csv` | Image IDs, coordinates, bounding boxes |
| `data/scores.csv` | Per-image walkability scores |
| `data/area_summary.csv` | Area-level mean/std walkability |
| `data/osm_scores.csv` | OSM-derived metrics per area |
| `data/validation.txt` | Classification report |
| `walkability_map.html` | Interactive map |
| `plot_boxplot.png` | Score distributions per area |
| `plot_barchart.png` | SegFormer vs OSM feature comparison |
| `plot_distributions.png` | Per-image KDE distributions vs OSM |

---

## Scoring Formula

```
raw_score = 0.6 × R_sidewalk + 0.4 × R_vegetation − 1.5 × R_obstruction
walkability = min-max normalise(raw_score) → [0, 1]
```

ADE20K classes used:
- Sidewalk: class 11
- Vegetation: classes 4 (tree), 9 (grass), 17 (plant), 72 (palm)
- Obstruction: class 32 (fence)
