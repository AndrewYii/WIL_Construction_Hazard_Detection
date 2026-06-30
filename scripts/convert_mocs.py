"""
Convert MOCS dataset (COCO-style instances_*.json) to our 2-class YOLO format.

MOCS has 13 classes. Mapping:
  worker (0):           person, worker
  dangerous_vehicle (1): truck, crane, excavator, roller, bulldozer, loader,
                          concrete mixer truck, vehicle (catch-all)
  drop:                 hardhat, vest, cone, sign, barrier, scaffold

Annotation files: instances_train.json, instances_val.json, instances_test.json
Images: train/, val/, test/ subdirs (confirm paths after image archives are unpacked)

Usage:
    python scripts/convert_mocs.py \
        --src data/raw/mocs \
        --dst data/raw/mocs/yolo_labels
"""

import argparse
import json
from pathlib import Path


WORKER_CLASSES = {
    "person", "worker", "pedestrian", "human"
}

VEHICLE_CLASSES = {
    "truck", "crane", "excavator", "roller", "bulldozer", "loader",
    "wheel loader", "forklift", "backhoe", "concrete mixer", "concrete mixer truck",
    "vehicle", "car", "van", "bus", "dump truck", "machinery", "machine"
}


def remap(name: str) -> int | None:
    n = name.lower().strip()
    if n in WORKER_CLASSES:
        return 0
    if n in VEHICLE_CLASSES:
        return 1
    return None


def coco_bbox_to_yolo(bbox, w, h):
    x, y, bw, bh = bbox
    return (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h


def load_bad_images(src: Path) -> set[str]:
    """Load stems of corrupt/missing images flagged by download_mocs.py."""
    bad_file = src / "bad_images.txt"
    if not bad_file.exists():
        return set()
    stems = set()
    for line in bad_file.read_text().splitlines():
        line = line.strip()
        if line:
            stems.add(Path(line).stem)
    if stems:
        print(f"  Skipping {len(stems)} bad images listed in bad_images.txt")
    return stems


def convert(json_path: Path, label_dir: Path, bad_stems: set[str] = None):
    with open(json_path) as f:
        coco = json.load(f)

    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    id_to_img = {img["id"]: img for img in coco["images"]}

    anns_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    label_dir.mkdir(parents=True, exist_ok=True)
    skipped_ann = skipped_img = 0

    for img_id, img in id_to_img.items():
        stem = Path(img["file_name"]).stem
        if bad_stems and stem in bad_stems:
            skipped_img += 1
            continue
        lines = []
        for ann in anns_by_img.get(img_id, []):
            cls = remap(id_to_name.get(ann["category_id"], ""))
            if cls is None:
                skipped_ann += 1
                continue
            cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], img["width"], img["height"])
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (label_dir / f"{stem}.txt").write_text("\n".join(lines))

    print(f"  {json_path.name}: {len(id_to_img)} images | {skipped_img} bad skipped | {skipped_ann} annotations dropped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    bad_stems = load_bad_images(src)

    # MOCS val: annotations_val.json, images in instances_val/
    # MOCS test: image_info_test.json has NO annotations — skip
    split_configs = [
        ("val", ["annotations_val.json", "instances_val.json", "annotations/instances_val.json"]),
    ]

    for split, candidates in split_configs:
        jf = next((src / c for c in candidates if (src / c).exists()), None)
        if jf is None:
            print(f"  {split}: annotation file not found, skipping")
            continue
        print(f"Processing {split} (from {jf.name}) ...")
        convert(jf, dst / split, bad_stems)

    print("Done.")


if __name__ == "__main__":
    main()
