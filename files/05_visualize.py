"""
04_visualize.py — Generate all walkability visualizations.

Produces:
  - walkability_map.html  : Interactive Folium map with colour-coded markers
  - plot_boxplot.png      : Walkability score distribution per area
  - plot_barchart.png     : SegFormer vs OSM feature comparison
  - plot_distributions.png: Per-image score distributions vs OSM values

Requires scores.csv (from 02_segment.py) and OSM data (from 05_osm_validate.py).

Usage:
    python 04_visualize.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from scipy.stats import gaussian_kde

from config import (
    SCORES_CSV, SUMMARY_CSV, AREA_LABELS,
    AREA_ORDER, THRESHOLD_HIGH, THRESHOLD_LOW,
)

OSM_CSV = "data/osm_scores.csv"  # produced by 05_osm_validate.py


# ========================
# HELPERS
# ========================

def load_data():
    df_scores = pd.read_csv(SCORES_CSV)
    df_scores["area"] = df_scores["bbox"].map(AREA_LABELS)
    df_summary = pd.read_csv(SUMMARY_CSV)

    try:
        df_osm = pd.read_csv(OSM_CSV)
    except FileNotFoundError:
        print("Warning: osm_scores.csv not found. OSM plots will be skipped.")
        df_osm = None

    return df_scores, df_summary, df_osm


def classify_score(score: float) -> str:
    if score >= THRESHOLD_HIGH:
        return "high"
    elif score >= THRESHOLD_LOW:
        return "medium"
    return "low"


# ========================
# MAP
# ========================

AREA_CENTROIDS = {
    "Rigga":               (25.277, 55.322),
    "Majaz":               (25.337, 55.387),
    "Al Khan":             (25.337, 55.357),
    "Deira":               (25.275, 55.315),
    "Dubai Downtown":      (25.205, 55.285),
    "Sharjah Residential": (25.315, 55.385),
    "Ajman":               (25.415, 55.480),
    "Industrial":          (25.325, 55.435),
    "Park":                (25.245, 55.295),
    "Reem Island":         (24.510, 54.440),
}

COLOR_MAP = {"high": "green", "medium": "orange", "low": "red"}


def build_map(df_summary: pd.DataFrame) -> folium.Map:
    m = folium.Map(location=[25.1, 55.2], zoom_start=9,
                   tiles="CartoDB positron")

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:40px;left:40px;z-index:1000;
         background:white;padding:12px;border-radius:8px;
         border:1px solid #ccc;font-size:13px;">
      <b>Walkability Score</b><br>
      <span style="color:green">●</span> High  (≥ 0.53)<br>
      <span style="color:orange">●</span> Medium (0.47–0.53)<br>
      <span style="color:red">●</span> Low   (&lt; 0.47)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    for _, row in df_summary.iterrows():
        area = row["area"]
        score = row["avg_walkability"]
        cat = classify_score(score)
        color = COLOR_MAP[cat]
        lat, lon = AREA_CENTROIDS.get(area, (25.2, 55.3))

        popup_html = f"""
        <b>{area}</b><br>
        Walkability: <b>{score:.3f}</b><br>
        Sidewalk coverage: {row['avg_sidewalk']:.3f}<br>
        Vegetation coverage: {row['avg_vegetation']:.3f}<br>
        Images: {int(row['n_images'])}
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=20,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{area}: {score:.3f}",
        ).add_to(m)

    return m


# ========================
# BOX PLOT
# ========================

def plot_boxplot(df_scores: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))

    data = [
        df_scores[df_scores["area"] == area]["walkability"].dropna().values
        for area in AREA_ORDER
    ]

    bp = ax.boxplot(data, patch_artist=True, vert=True)

    for patch, area in zip(bp["boxes"], AREA_ORDER):
        median = df_scores[df_scores["area"] == area]["walkability"].median()
        if median >= THRESHOLD_HIGH:
            patch.set_facecolor("#90EE90")
        elif median >= THRESHOLD_LOW:
            patch.set_facecolor("#FFD580")
        else:
            patch.set_facecolor("#FFB3B3")

    ax.axhline(THRESHOLD_HIGH, color="green",  linestyle="--", linewidth=1, label=f"High threshold ({THRESHOLD_HIGH})")
    ax.axhline(THRESHOLD_LOW,  color="red",    linestyle="--", linewidth=1, label=f"Low threshold ({THRESHOLD_LOW})")

    ax.set_xticks(range(1, len(AREA_ORDER) + 1))
    ax.set_xticklabels(AREA_ORDER, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Walkability Score")
    ax.set_title("Walkability Score Distribution by Area")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("plot_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plot_boxplot.png")


# ========================
# BAR CHART (SegFormer vs OSM)
# ========================

def plot_barchart(df_summary: pd.DataFrame, df_osm: pd.DataFrame):
    if df_osm is None:
        print("Skipping barchart: OSM data not available.")
        return

    df_merged = pd.merge(df_summary, df_osm, on="area")
    areas = df_merged["area"].tolist()
    x = np.arange(len(areas))
    width = 0.35

    features = [
        ("avg_sidewalk",    "osm_sidewalk_norm",  "Sidewalk"),
        ("avg_vegetation",  "osm_veg_norm",       "Vegetation"),
        ("avg_obstruction", "osm_barrier_norm",   "Obstruction"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, (seg_col, osm_col, label) in zip(axes, features):
        ax.bar(x - width/2, df_merged[seg_col], width, label="SegFormer", alpha=0.8)
        ax.bar(x + width/2, df_merged[osm_col],  width, label="OSM",       alpha=0.8)
        ax.set_ylabel("Normalised Score")
        ax.set_title(label)
        ax.legend(fontsize=9)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(areas, rotation=30, ha="right", fontsize=9)
    plt.suptitle("SegFormer vs OSM Feature Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plot_barchart.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plot_barchart.png")


# ========================
# DISTRIBUTION PLOT
# ========================

def plot_distributions(df_scores: pd.DataFrame, df_osm: pd.DataFrame):
    if df_osm is None:
        print("Skipping distribution plot: OSM data not available.")
        return

    df_classes = df_osm.copy()
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    components = [
        ("r_sidewalk",    "osm_sidewalk_norm", "Sidewalk",    "#4C9BE8"),
        ("r_vegetation",  "osm_veg_norm",      "Vegetation",  "#5DBB63"),
        ("r_obstruction", "osm_barrier_norm",  "Obstruction", "#E8694C"),
    ]

    for ax, (seg_col, osm_col, label, color) in zip(axes, components):
        for i, area in enumerate(AREA_ORDER):
            area_data = df_scores[df_scores["area"] == area][seg_col].dropna().values

            if len(area_data) > 1:
                kde_x = np.linspace(0, area_data.max() * 1.3 + 0.01, 300)
                kde   = gaussian_kde(area_data, bw_method=0.3)
                kde_y = kde(kde_x)
                kde_y = kde_y / kde_y.max() * 0.8
                ax.fill_between(kde_x, i, i + kde_y, alpha=0.4, color=color)
                ax.plot(kde_x, i + kde_y, color=color, linewidth=1)

            osm_val = df_classes.loc[df_classes["area"] == area, osm_col].values
            if len(osm_val) > 0:
                ax.vlines(osm_val[0], i, i + 0.8, color="black",
                          linewidth=1.8, linestyle="--", zorder=5,
                          label="OSM value" if i == 0 else None)
                ax.scatter(osm_val[0], i + 0.4, marker="D",
                           color="black", s=40, zorder=6)

        ax.set_yticks(range(len(AREA_ORDER)))
        ax.set_yticklabels(AREA_ORDER, fontsize=9)
        ax.set_xlabel(f"{label} Score (normalised)", labelpad=8)
        ax.set_title(f"{label} — SegFormer Distribution vs OSM Value (◆)",
                     fontsize=11, fontweight="bold")
        ax.set_xlim(0, None)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9, frameon=False, loc="upper right")

    plt.suptitle("Per-Image SegFormer Score Distributions vs OSM Ground Truth",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("plot_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plot_distributions.png")


# ========================
# MAIN
# ========================

def main():
    df_scores, df_summary, df_osm = load_data()

    # Map
    m = build_map(df_summary)
    m.save("walkability_map.html")
    print("Saved walkability_map.html")

    # Plots
    plot_boxplot(df_scores)
    plot_barchart(df_summary, df_osm)
    plot_distributions(df_scores, df_osm)


if __name__ == "__main__":
    main()
