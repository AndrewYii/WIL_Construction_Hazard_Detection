"""
Convert ConstructionSite10k (Parquet format) to YOLO format.

Dataset columns with bounding boxes (normalized [x1, y1, x2, y2]):
  excavator                -> dangerous_vehicle (1)
  worker_with_white_hard_hat -> worker (0)

rule_*_violation columns are skipped (proximity/rule logic, not pure class boxes).
rebar is skipped (out of scope).

Usage:
    python scripts/convert_constructionsite10k.py \
        --src data/raw/constructionsite10k \
        --dst data/raw/constructionsite10k/yolo_labels
"""

import argparse
from pathlib import Path
import numpy as np


COLUMN_MAP = {
    "excavator": 1,
    "worker_with_white_hard_hat": 0,
}


def xyxy_to_yolo(box, img_w=1.0, img_h=1.0):
    """Convert [x1, y1, x2, y2] normalised to YOLO [cx, cy, w, h] normalised."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


def extract_boxes(cell) -> list:
    """Extract list of bounding boxes from a cell (array or dict)."""
    if cell is None:
        return []
    if isinstance(cell, dict):
        raw = cell.get("bounding_box", [])
    else:
        raw = cell
    boxes = []
    for item in raw:
        if item is None:
            continue
        arr = np.asarray(item).flatten()
        if len(arr) == 4:
            boxes.append(arr.tolist())
    return boxes


def convert(src: Path, dst: Path):
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas not installed. Run: pip install pandas pyarrow")

    img_dir = dst / "images"
    lbl_dir = dst / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(src.glob("*.parquet"))
    if not parquet_files:
        print("No parquet files found under", src)
        return

    total = skipped = 0

    for pf in parquet_files:
        print(f"  Processing {pf.name} ...")
        df = pd.read_parquet(pf)

        for _, row in df.iterrows():
            img_id = row["image_id"]
            lines = []

            for col, cls in COLUMN_MAP.items():
                if col not in row:
                    continue
                for box in extract_boxes(row[col]):
                    cx, cy, w, h = xyxy_to_yolo(box)
                    if w <= 0 or h <= 0:
                        continue
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Save image
            img_bytes = row["image"]
            if isinstance(img_bytes, dict):
                img_bytes = img_bytes.get("bytes", b"")
            if img_bytes:
                img_path = img_dir / f"{img_id}.jpg"
                img_path.write_bytes(img_bytes)

            # Save label
            lbl_path = lbl_dir / f"{img_id}.txt"
            lbl_path.write_text("\n".join(lines))

            if lines:
                total += 1
            else:
                skipped += 1

    print(f"  Done: {total} images with annotations | {skipped} images with no relevant objects")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    convert(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
