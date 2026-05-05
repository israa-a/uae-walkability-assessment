"""
config.py — Central configuration for the UAE walkability pipeline.
Edit paths and settings here before running any module.
"""

# ========================
# MAPILLARY
# ========================
ACCESS_TOKEN = "YOUR_MAPILLARY_TOKEN_HERE"  # Replace with your token
MAX_PER_BBOX = 200  # Max images to download per area

# ========================
# PATHS
# ========================
IMAGE_DIR   = "data/images"          # Where downloaded images are stored
METADATA_CSV = "data/metadata.csv"   # Output of 01_download.py
SCORES_CSV   = "data/scores.csv"     # Output of 02_segment.py
SUMMARY_CSV  = "data/area_summary.csv"  # Output of 03_aggregate.py

# ========================
# BOUNDING BOXES
# ========================
BBOXES = [
    "55.270,25.190,55.300,25.220",  # Dubai Downtown
    "55.300,25.260,55.330,25.290",  # Deira
    "55.370,25.300,55.400,25.330",  # Sharjah Residential
    "55.420,25.310,55.450,25.340",  # Industrial
    "55.280,25.230,55.310,25.260",  # Park (Zabeel/Mankhool)
    "55.315,25.270,55.330,25.285",  # Rigga
    "55.380,25.330,55.395,25.345",  # Majaz
    "55.350,25.330,55.365,25.345",  # Al Khan
    "55.470,25.405,55.490,25.425",  # Ajman
    "54.430,24.500,54.450,24.520",  # Reem Island
]

AREA_LABELS = {
    "55.270,25.190,55.300,25.220": "Dubai Downtown",
    "55.300,25.260,55.330,25.290": "Deira",
    "55.370,25.300,55.400,25.330": "Sharjah Residential",
    "55.420,25.310,55.450,25.340": "Industrial",
    "55.280,25.230,55.310,25.260": "Park",
    "55.315,25.270,55.330,25.285": "Rigga",
    "55.380,25.330,55.395,25.345": "Majaz",
    "55.350,25.330,55.365,25.345": "Al Khan",
    "55.470,25.405,55.490,25.425": "Ajman",
    "54.430,24.500,54.450,24.520": "Reem Island",
}

# Display order for plots (best to worst walkability)
AREA_ORDER = [
    "Rigga", "Majaz", "Al Khan", "Deira", "Dubai Downtown",
    "Sharjah Residential", "Ajman", "Industrial", "Park", "Reem Island"
]

# ========================
# SEGMENTATION MODEL
# ========================
MODEL_CHECKPOINT = "nvidia/segformer-b2-finetuned-ade-512-512"

# ADE20K class indices used in scoring
ADE20K_SIDEWALK    = {11}           # sidewalk / pavement
ADE20K_VEGETATION  = {4, 9, 17, 72} # tree, grass, plant, palm
ADE20K_OBSTRUCTION = {32}           # fence

# ========================
# SCORING WEIGHTS
# ========================
W_SIDEWALK    =  0.6
W_VEGETATION  =  0.4
W_OBSTRUCTION = -1.5

# ========================
# WALKABILITY THRESHOLDS
# ========================
THRESHOLD_HIGH = 0.53
THRESHOLD_LOW  = 0.47
