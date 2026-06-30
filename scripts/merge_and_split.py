"""
Merge converted YOLO datasets into data/processed/ with train and test splits only.

Expects each source to have:
  <source_dir>/
    images/   (or alongside label dir)
    yolo_labels/   (converted .txt files)

Sources:
  - ConstructionSite10k           -> train pool
  - Roboflow v27                  -> train pool
  - Construction-Hazard-Detection -> train pool
  - MOCS val set                  -> train pool (only labeled split available)

MOCS test set has no annotations — excluded entirely.
Test split is carved from the merged pool (--test-ratio, default 0.15).
No validation split.

Usage:
    python scripts/merge_and_split.py \
        --sources data/raw/constructionsite10k data/raw/roboflow_v27 data/raw/hazard_detection \
        --mocs-dir data/raw/mocs \
        --dst data/processed \
        [--test-ratio 0.15] [--seed 42]
"""

import argparse
import random
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_image(label_path: Path, img_roots: list[Path]) -> Path | None:
    stem = label_path.stem
    for root in img_roots:
        for ext in IMG_EXTS:
            candidate = root / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def collect_pairs(label_dir: Path, img_roots: list[Path]) -> list[tuple[Path, Path]]:
    pairs = []
    for lf in label_dir.rglob("*.txt"):
        if lf.stat().st_size == 0:
            continue
        img = find_image(lf, img_roots)
        if img is None:
            continue
        pairs.append((img, lf))
    return pairs


def copy_pair(img: Path, lbl: Path, img_dst: Path, lbl_dst: Path, prefix: str = ""):
    name = f"{prefix}{img.stem}" if prefix else img.stem
    shutil.copy2(img, img_dst / f"{name}{img.suffix}")
    shutil.copy2(lbl, lbl_dst / f"{name}.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True, help="Converted source dirs (non-MOCS)")
    parser.add_argument("--mocs-dir", default=None, help="Path to converted MOCS dir")
    parser.add_argument("--dst", required=True, help="data/processed")
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dst = Path(args.dst)
    for split in ("train", "test"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    pool: list[tuple[Path, Path]] = []

    # --- non-MOCS sources -> training pool ---
    for src_str in args.sources:
        src = Path(src_str)
        label_dir = src / "yolo_labels"
        if not label_dir.exists():
            label_dir = src / "labels"
        if not label_dir.exists():
            print(f"  WARNING: no yolo_labels/ or labels/ found, skipping {src.name}")
            continue
        img_roots = [
            src / "images",
            src / "train" / "images",
            src / "valid" / "images",
            src / "test" / "images",
            src,
        ]
        pairs = collect_pairs(label_dir, img_roots)
        print(f"  {src.name}: {len(pairs)} valid pairs -> pool")
        pool.extend(pairs)

    # --- MOCS val -> training pool (test set has no annotations, excluded) ---
    if args.mocs_dir:
        mocs = Path(args.mocs_dir)
        val_ldir = mocs / "yolo_labels" / "val"
        if val_ldir.exists():
            img_roots = [mocs / "instances_val", mocs / "val", mocs / "images" / "val", mocs]
            pairs = collect_pairs(val_ldir, img_roots)
            print(f"  MOCS val: {len(pairs)} pairs -> pool")
            pool.extend(pairs)
        else:
            print("  MOCS val: label dir not found, skipping")

    if not pool:
        print("No pairs found. Check that conversion scripts have been run first.")
        return

    random.seed(args.seed)
    random.shuffle(pool)

    n_test = max(1, int(len(pool) * args.test_ratio))
    test_pairs = pool[:n_test]
    train_pairs = pool[n_test:]

    print(f"\nTotal pool: {len(pool)} | train: {len(train_pairs)} | test: {len(test_pairs)}")

    for img, lbl in train_pairs:
        copy_pair(img, lbl, dst / "images" / "train", dst / "labels" / "train")

    for img, lbl in test_pairs:
        copy_pair(img, lbl, dst / "images" / "test", dst / "labels" / "test")

    print("Merge complete.")
    print(f"  processed/images/train : {len(list((dst/'images'/'train').iterdir()))}")
    print(f"  processed/images/test  : {len(list((dst/'images'/'test').iterdir()))}")


if __name__ == "__main__":
    main()
