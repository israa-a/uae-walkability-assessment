"""
02_segment.py — Run SegFormer-B2 segmentation and compute walkability scores.

Reads images from IMAGE_DIR and metadata from METADATA_CSV.
For each image, runs semantic segmentation and computes pixel coverage
ratios for sidewalk, vegetation, and obstruction. Applies weighted scoring
and min-max normalisation. Saves per-image scores to SCORES_CSV.

Usage:
    python 02_segment.py
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from config import (
    IMAGE_DIR, METADATA_CSV, SCORES_CSV, MODEL_CHECKPOINT,
    ADE20K_SIDEWALK, ADE20K_VEGETATION, ADE20K_OBSTRUCTION,
    W_SIDEWALK, W_VEGETATION, W_OBSTRUCTION,
)


def load_model(checkpoint: str, device: str):
    """Load SegFormer processor and model onto the specified device."""
    print(f"Loading model: {checkpoint}")
    processor = SegformerImageProcessor.from_pretrained(checkpoint)
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint)
    model.eval()
    model.to(device)
    return processor, model


def segment_image(image: Image.Image, processor, model, device: str) -> np.ndarray:
    """
    Run segmentation on a PIL image.
    Returns a 2D numpy array of ADE20K class indices (same size as input image).
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    upsampled = torch.nn.functional.interpolate(
        outputs.logits,
        size=image.size[::-1],  # (height, width)
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1).squeeze().cpu().numpy()


def compute_pixel_ratios(seg_map: np.ndarray) -> tuple[float, float, float]:
    """
    Compute pixel coverage ratios for sidewalk, vegetation, and obstruction.
    Returns (r_sidewalk, r_vegetation, r_obstruction).
    """
    total = seg_map.size
    r_sidewalk = np.isin(seg_map, list(ADE20K_SIDEWALK)).sum()    / total
    r_veg      = np.isin(seg_map, list(ADE20K_VEGETATION)).sum()  / total
    r_obs      = np.isin(seg_map, list(ADE20K_OBSTRUCTION)).sum() / total
    return r_sidewalk, r_veg, r_obs


def compute_raw_score(r_sidewalk: float, r_veg: float, r_obs: float) -> float:
    """Apply weighted scoring formula."""
    return W_SIDEWALK * r_sidewalk + W_VEGETATION * r_veg + W_OBSTRUCTION * r_obs


def normalize_scores(series: pd.Series) -> pd.Series:
    """Min-max normalise a pandas Series to [0, 1]."""
    mn, mx = series.min(), series.max()
    return ((series - mn) / (mx - mn)).round(4)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor, model = load_model(MODEL_CHECKPOINT, device)

    df_meta = pd.read_csv(METADATA_CSV)
    df_meta["image_id"] = df_meta["image_id"].astype(str)

    results = []
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]

    for fname in tqdm(image_files, desc="Segmenting images"):
        image_id = fname.replace(".jpg", "")
        img_path = os.path.join(IMAGE_DIR, fname)

        try:
            image = Image.open(img_path).convert("RGB")
            seg_map = segment_image(image, processor, model, device)
            r_sw, r_veg, r_obs = compute_pixel_ratios(seg_map)
            raw = compute_raw_score(r_sw, r_veg, r_obs)

            results.append({
                "image_id":      image_id,
                "r_sidewalk":    round(r_sw,  4),
                "r_vegetation":  round(r_veg, 4),
                "r_obstruction": round(r_obs, 4),
                "raw_score":     round(raw,   4),
            })

        except Exception as e:
            print(f"  Failed: {fname} — {e}")

    df_scores = pd.DataFrame(results)
    df_scores["walkability"] = normalize_scores(df_scores["raw_score"])
    df_scores["image_id"] = df_scores["image_id"].astype(str)

    # Merge with metadata to attach coordinates and bbox
    df_final = pd.merge(
        df_scores,
        df_meta[["image_id", "longitude", "latitude", "bbox"]],
        on="image_id",
        how="left",
    )

    os.makedirs(os.path.dirname(SCORES_CSV), exist_ok=True)
    df_final.to_csv(SCORES_CSV, index=False)
    print(f"\nDone. Scores saved to {SCORES_CSV}")
    print(df_final.describe())


if __name__ == "__main__":
    main()
