"""
Convert Construction-Hazard-Detection (Roboflow) to our 2-class YOLO format.

This dataset is closest to an explicit proximity class. Class names vary by export version;
run with --inspect first to see what classes your download contains, then adjust the map.

Usage:
    # First inspect class names in your download:
    python scripts/convert_hazard_detection.py --src data/raw/hazard_detection --inspect

    # Then convert:
    python scripts/convert_hazard_detection.py \
        --src data/raw/hazard_detection \
        --dst data/raw/hazard_detection/yolo_labels
"""

import argparse
import re
from pathlib import Path


WORKER_KEYWORDS = {"person", "worker", "human", "people", "pedestrian"}
VEHICLE_KEYWORDS = {
    "vehicle", "car", "truck", "van", "excavator", "crane", "bulldozer",
    "loader", "forklift", "machinery", "machine", "roller", "backhoe"
}


def classify_name(name: str) -> int | None:
    n = name.lower().strip()
    if any(k in n for k in WORKER_KEYWORDS):
        return 0
    if any(k in n for k in VEHICLE_KEYWORDS):
        return 1
    return None


def find_classes_file(src: Path) -> Path | None:
    for name in ("obj.names", "classes.txt", "_darknet.labels"):
        f = next(src.rglob(name), None)
        if f:
            return f
    data_yaml = next(src.rglob("data.yaml"), None)
    return data_yaml


def parse_class_names(path: Path) -> list[str]:
    if path.suffix == ".yaml":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, list):
            return [str(n) for n in names]
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def build_map(class_names: list[str]) -> dict[int, int]:
    mapping = {}
    for i, name in enumerate(class_names):
        cls = classify_name(name)
        if cls is not None:
            mapping[i] = cls
    return mapping


def convert(src: Path, dst: Path, class_map: dict[int, int]):
    txt_files = [
        f for f in src.rglob("*.txt")
        if f.name not in {"classes.txt", "obj.names", "_darknet.labels", "README.dataset.txt", "README.roboflow.txt"}
        and "labels" in f.parts
    ]
    dst.mkdir(parents=True, exist_ok=True)
    converted = skipped_files = 0
    for lf in txt_files:
        lines_out = []
        for line in lf.read_text().splitlines():
            parts = line.strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            mapped = class_map.get(int(parts[0]))
            if mapped is None:
                continue
            lines_out.append(f"{mapped} {' '.join(parts[1:])}")
        out = dst / lf.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines_out))
        if lines_out:
            converted += 1
        else:
            skipped_files += 1
    print(f"Converted: {converted} | empty after remap: {skipped_files}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", default=None)
    parser.add_argument("--inspect", action="store_true", help="Print class names and exit")
    args = parser.parse_args()

    src = Path(args.src)
    classes_file = find_classes_file(src)
    if not classes_file:
        print("Could not find a class names file. Expected obj.names, classes.txt, or data.yaml.")
        return

    class_names = parse_class_names(classes_file)
    class_map = build_map(class_names)

    print("Class mapping:")
    for i, name in enumerate(class_names):
        mapped = class_map.get(i, "DROP")
        label = {0: "worker", 1: "dangerous_vehicle"}.get(mapped, "drop")
        print(f"  [{i}] {name}  ->  {label}")

    if args.inspect:
        return

    if not args.dst:
        print("Pass --dst to run conversion.")
        return

    convert(src, Path(args.dst), class_map)
    print("Done.")


if __name__ == "__main__":
    main()
