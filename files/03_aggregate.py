"""
03_aggregate.py — Aggregate per-image scores to area level.

Reads SCORES_CSV, maps bounding boxes to area names, computes
mean and std walkability per area, and saves SUMMARY_CSV.

Usage:
    python 03_aggregate.py
"""

import pandas as pd

from config import SCORES_CSV, SUMMARY_CSV, AREA_LABELS


def aggregate(scores_csv: str, area_labels: dict) -> pd.DataFrame:
    """Load scores, attach area labels, and return area-level summary."""
    df = pd.read_csv(scores_csv)
    df["area"] = df["bbox"].map(area_labels)

    summary = df.groupby("area").agg(
        n_images        =("image_id",      "count"),
        avg_sidewalk    =("r_sidewalk",     "mean"),
        avg_vegetation  =("r_vegetation",   "mean"),
        avg_obstruction =("r_obstruction",  "mean"),
        avg_walkability =("walkability",    "mean"),
        std_walkability =("walkability",    "std"),
    ).reset_index().round(4)

    return summary.sort_values("avg_walkability", ascending=False)


def main():
    summary = aggregate(SCORES_CSV, AREA_LABELS)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
