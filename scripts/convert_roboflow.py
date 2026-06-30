"""
Convert Roboflow Construction Site Safety v27 to our 2-class YOLO format.

Roboflow exports come as YOLO-format .txt labels already, but with their own class indices
defined in data.yaml or obj.data. We re-map them to our taxonomy:
  0 = worker
  1 = dangerous_vehicle

Roboflow v27 known classes (spot-check "machinery" before committing to dangerous_vehicle):
  0: Hardhat       -> drop
  1: Mask          -> drop
  2: NO-Hardhat    -> drop
  3: NO-Mask       -> drop
  4: NO-Safety Vest-> drop
  5: Person        -> worker (0)
  6: Safety Cone   -> drop
  7: Safety Vest   -> drop
  8: machinery     -> dangerous_vehicle (1)  *** spot-check advised ***
  9: vehicle       -> dangerous_vehicle (1)

Usage:
    python scripts/convert_roboflow.py \
        --src data/raw/roboflow_v27 \
        --dst data/raw/roboflow_v27/yolo_labels \
        [--class-map path/to/class_map.json]

If --class-map is not provided, the hardcoded mapping above is used.
Pass --class-map to override if your export has different class indices.
"""

import argparse
import json
import os
import shutil
from pathlib import Path


DEFAULT_CLASS_MAP = {
    5: 0,   # Person -> worker
    8: 1,   # machinery -> dangerous_vehicle  (spot-check advised)
    9: 1,   # vehicle -> dangerous_vehicle
}


def load_class_map(path: str | None) -> dict[int, int]:
    if path:
        with open(path) as f:
            raw = json.load(f)
        return {int(k): int(v) for k, v in raw.items()}
    return DEFAULT_CLASS_MAP


def convert_label_file(src_file: Path, dst_file: Path, class_map: dict[int, int]):
    lines_out = []
    with open(src_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            original_cls = int(parts[0])
            mapped = class_map.get(original_cls)
            if mapped is None:
                continue
            lines_out.append(f"{mapped} {' '.join(parts[1:])}")
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    dst_file.write_text("\n".join(lines_out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--class-map", default=None, help="Optional JSON file: {old_idx: new_idx}")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    class_map = load_class_map(args.class_map)

    txt_files = [
        f for f in src.rglob("*.txt")
        if f.name not in {"classes.txt", "obj.names", "README.dataset.txt", "README.roboflow.txt"}
        and "labels" in f.parts
    ]
    if not txt_files:
        print("No label .txt files found under", src)
        return

    converted = skipped = 0
    for lf in txt_files:
        relative = lf.relative_to(src)
        out = dst / relative
        convert_label_file(lf, out, class_map)
        with open(out) as f:
            if f.read().strip():
                converted += 1
            else:
                skipped += 1

    print(f"Converted: {converted} files with kept annotations | {skipped} files became empty (all out-of-scope)")
    print("NOTE: 'machinery' class was mapped to dangerous_vehicle — run a spot-check before training.")


if __name__ == "__main__":
    main()
