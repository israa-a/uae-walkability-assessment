"""
01_download.py — Download street-level images from Mapillary.

For each bounding box defined in config.py, queries the Mapillary API,
randomly samples up to MAX_PER_BBOX images, downloads them, and saves
metadata (image_id, longitude, latitude, bbox) to METADATA_CSV.

Usage:
    python 01_download.py
"""

import os
import time
import random

import requests
import pandas as pd
from tqdm import tqdm

from config import (
    ACCESS_TOKEN, BBOXES, MAX_PER_BBOX,
    IMAGE_DIR, METADATA_CSV
)


def fetch_image_metadata(bbox: str, access_token: str) -> list[dict]:
    """Fetch all image metadata for a bounding box from the Mapillary API."""
    url = (
        f"https://graph.mapillary.com/images"
        f"?bbox={bbox}"
        f"&fields=id,thumb_2048_url,computed_geometry"
        f"&access_token={access_token}"
    )
    all_images = []
    while url:
        response = requests.get(url).json()
        all_images.extend(response.get("data", []))
        url = response.get("paging", {}).get("next")
        time.sleep(0.3)
    return all_images


def download_image(img: dict, image_dir: str) -> dict | None:
    """Download a single image and return its metadata, or None on failure."""
    img_id  = img["id"]
    img_url = img.get("thumb_2048_url")
    geometry = img.get("computed_geometry", {})

    if not img_url:
        return None

    try:
        img_data = requests.get(img_url).content
        out_path = os.path.join(image_dir, f"{img_id}.jpg")
        with open(out_path, "wb") as f:
            f.write(img_data)

        coords = geometry.get("coordinates", [None, None])
        return {
            "image_id":  img_id,
            "longitude": coords[0],
            "latitude":  coords[1],
        }
    except Exception as e:
        print(f"  Failed to download {img_id}: {e}")
        return None


def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_CSV), exist_ok=True)

    metadata_list = []
    downloaded_ids = set()

    for bbox in BBOXES:
        print(f"\nProcessing bbox: {bbox}")

        all_images = fetch_image_metadata(bbox, ACCESS_TOKEN)
        print(f"  Total available: {len(all_images)}")

        sampled = random.sample(all_images, min(len(all_images), MAX_PER_BBOX))
        print(f"  Downloading: {len(sampled)} images")

        for img in tqdm(sampled):
            if img["id"] in downloaded_ids:
                continue
            downloaded_ids.add(img["id"])

            meta = download_image(img, IMAGE_DIR)
            if meta:
                meta["bbox"] = bbox
                metadata_list.append(meta)

    df = pd.DataFrame(metadata_list)
    df.to_csv(METADATA_CSV, index=False)
    print(f"\nDone. {len(downloaded_ids)} images downloaded.")
    print(f"Metadata saved to {METADATA_CSV}")


if __name__ == "__main__":
    main()
