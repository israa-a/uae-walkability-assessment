"""
05_osm_validate.py — Fetch OSM data and compute validation scores.

For each study area bounding box, retrieves pedestrian infrastructure,
vegetation, and barrier features from OpenStreetMap using OSMnx.
Computes normalised OSM composite scores using the same weighted formula
as the SegFormer pipeline, then evaluates classification agreement.

Outputs:
  - data/osm_scores.csv    : Per-area OSM metrics and composite scores
  - data/validation.txt    : Classification report and accuracy

Usage:
    python 05_osm_validate.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import osmnx as ox
from sklearn.metrics import classification_report, accuracy_score

from config import (
    BBOXES, AREA_LABELS, SCORES_CSV, SUMMARY_CSV,
    W_SIDEWALK, W_VEGETATION, W_OBSTRUCTION,
)

OSM_CSV        = "data/osm_scores.csv"
VALIDATION_TXT = "data/validation.txt"

# OSM tag filters
PEDESTRIAN_HIGHWAY_TAGS = {"footway", "pedestrian", "steps", "living_street"}

VEGETATION_TAGS = {
    "landuse": ["grass", "village_green", "flowerbed"],
    "natural": ["tree"],
    "leisure": ["garden", "park"],
}

BARRIER_TAGS = {
    "barrier": ["wall", "fence", "hedge", "jersey_barrier", "block"],
}


def bbox_to_tuple(bbox_str: str) -> tuple:
    """Convert '55.270,25.190,55.300,25.220' to (north, south, east, west)."""
    w, s, e, n = map(float, bbox_str.split(","))
    return n, s, e, w


def fetch_pedestrian_coverage(bbox_str: str) -> float:
    """Fraction of walkable edges tagged as pedestrian infrastructure."""
    try:
        n, s, e, w = bbox_to_tuple(bbox_str)
        G = ox.graph_from_bbox(n, s, e, w, network_type="walk")
        edges = ox.graph_to_gdfs(G, nodes=False)
        total = len(edges)
        if total == 0:
            return 0.0
        pedestrian = edges["highway"].apply(
            lambda h: bool(set([h] if isinstance(h, str) else h)
                           & PEDESTRIAN_HIGHWAY_TAGS)
        ).sum()
        return pedestrian / total
    except Exception as e:
        print(f"  Pedestrian fetch failed for {bbox_str}: {e}")
        return 0.0


def fetch_vegetation_count(bbox_str: str) -> int:
    """Count OSM features tagged with vegetation-related attributes."""
    n, s, e, w = bbox_to_tuple(bbox_str)
    total = 0
    for key, values in VEGETATION_TAGS.items():
        for val in values:
            try:
                gdf = ox.features_from_bbox(n, s, e, w, tags={key: val})
                total += len(gdf)
            except Exception:
                pass
    return total


def fetch_barrier_count(bbox_str: str) -> int:
    """Count OSM features tagged as physical obstructions (excluding gates)."""
    n, s, e, w = bbox_to_tuple(bbox_str)
    total = 0
    for key, values in BARRIER_TAGS.items():
        for val in values:
            try:
                gdf = ox.features_from_bbox(n, s, e, w, tags={key: val})
                total += len(gdf)
            except Exception:
                pass
    return total


def bbox_area_sq_deg(bbox_str: str) -> float:
    """Compute bounding box area in square degrees for density normalisation."""
    w, s, e, n = map(float, bbox_str.split(","))
    return (e - w) * (n - s)


def minmax_normalize(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0.0
    return (series - mn) / (mx - mn)


def classify_quantile(series: pd.Series) -> pd.Series:
    """Assign Low / Medium / High using tertile (quantile) binning."""
    q33 = series.quantile(1/3)
    q67 = series.quantile(2/3)
    return series.apply(
        lambda x: "High" if x > q67 else ("Medium" if x > q33 else "Low")
    )


def main():
    rows = []

    for bbox in BBOXES:
        area = AREA_LABELS[bbox]
        print(f"Fetching OSM data: {area}")

        area_sq_deg = bbox_area_sq_deg(bbox)

        sw_cov  = fetch_pedestrian_coverage(bbox)
        veg_cnt = fetch_vegetation_count(bbox) / area_sq_deg
        bar_cnt = fetch_barrier_count(bbox)    / area_sq_deg

        rows.append({
            "area":        area,
            "osm_sidewalk": sw_cov,
            "osm_veg_raw":  veg_cnt,
            "osm_bar_raw":  bar_cnt,
        })

    df = pd.DataFrame(rows)

    # Normalise
    df["osm_sidewalk_norm"] = minmax_normalize(df["osm_sidewalk"])
    df["osm_veg_norm"]      = minmax_normalize(df["osm_veg_raw"])
    df["osm_barrier_norm"]  = minmax_normalize(df["osm_bar_raw"])

    # OSM composite score (same formula as SegFormer)
    df["osm_raw_score"] = (
        W_SIDEWALK    * df["osm_sidewalk_norm"]
        + W_VEGETATION  * df["osm_veg_norm"]
        + W_OBSTRUCTION * df["osm_barrier_norm"]
    )
    df["osm_walkability"] = minmax_normalize(df["osm_raw_score"])

    df.to_csv(OSM_CSV, index=False)
    print(f"\nOSM scores saved to {OSM_CSV}")

    # ========================
    # CLASSIFICATION VALIDATION
    # ========================
    df_summary = pd.read_csv(SUMMARY_CSV)
    df_val = pd.merge(df, df_summary[["area", "avg_walkability"]], on="area")

    df_val["osm_class"] = classify_quantile(df_val["osm_walkability"])
    df_val["seg_class"] = classify_quantile(df_val["avg_walkability"])

    report = classification_report(
        df_val["osm_class"], df_val["seg_class"],
        labels=["Low", "Medium", "High"]
    )
    accuracy = accuracy_score(df_val["osm_class"], df_val["seg_class"])

    output = (
        f"Classification Report (SegFormer vs OSM ground truth)\n"
        f"{'='*55}\n"
        f"{report}\n"
        f"Overall accuracy: {accuracy:.2f} ({int(accuracy * len(df_val))}/{len(df_val)} areas)\n"
    )

    print(output)
    with open(VALIDATION_TXT, "w") as f:
        f.write(output)
    print(f"Validation results saved to {VALIDATION_TXT}")


if __name__ == "__main__":
    main()
