"""
Plan B — YOLO-World zero-shot evaluation against the same test set used for Plan A.

No training: just prompts the model with class names/synonyms and scores the
resulting boxes against our existing YOLO-format ground truth labels using a
simple IoU>=0.5 matching (precision/recall/F1 per class), so it's directly
comparable to Plan A's YOLOv8 numbers.

Usage:
    python scripts/eval_plan_b_yolow.py --n-samples 500
"""

import argparse
import json
import random
import time
from pathlib import Path

from ultralytics import YOLO

WORKER_PROMPTS = ["person", "worker", "construction worker"]
VEHICLE_PROMPTS = ["truck", "excavator", "crane", "bulldozer", "forklift", "vehicle", "loader"]
ALL_PROMPTS = WORKER_PROMPTS + VEHICLE_PROMPTS
PROMPT_TO_CLASS = {p: 0 for p in WORKER_PROMPTS}
PROMPT_TO_CLASS.update({p: 1 for p in VEHICLE_PROMPTS})


def load_gt(label_path: Path, img_w: int, img_h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append((cls, (x1, y1, x2, y2)))
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def match(preds, gts, iou_thresh=0.5):
    """preds/gts: list of (cls, box). Returns tp, fp, fn counts per class."""
    stats = {0: {"tp": 0, "fp": 0, "fn": 0}, 1: {"tp": 0, "fp": 0, "fn": 0}}
    matched_gt = set()
    for p_cls, p_box in preds:
        best_iou, best_idx = 0, -1
        for i, (g_cls, g_box) in enumerate(gts):
            if i in matched_gt or g_cls != p_cls:
                continue
            score = iou(p_box, g_box)
            if score > best_iou:
                best_iou, best_idx = score, i
        if best_iou >= iou_thresh:
            matched_gt.add(best_idx)
            stats[p_cls]["tp"] += 1
        else:
            stats[p_cls]["fp"] += 1
    for i, (g_cls, _) in enumerate(gts):
        if i not in matched_gt:
            stats[g_cls]["fn"] += 1
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/processed/images/test")
    parser.add_argument("--labels", default="data/processed/labels/test")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/plan_b_yolow_eval.json")
    args = parser.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    random.seed(args.seed)
    sample = random.sample(all_images, min(args.n_samples, len(all_images)))
    print(f"Evaluating on {len(sample)} / {len(all_images)} test images")

    model = YOLO("yolov8s-world.pt")
    model.set_classes(ALL_PROMPTS)

    totals = {0: {"tp": 0, "fp": 0, "fn": 0}, 1: {"tp": 0, "fp": 0, "fn": 0}}
    start = time.time()

    for i, img_path in enumerate(sample):
        result = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        img_h, img_w = result.orig_shape

        preds = []
        if result.boxes is not None:
            for box in result.boxes:
                prompt_idx = int(box.cls.item())
                prompt = ALL_PROMPTS[prompt_idx]
                cls = PROMPT_TO_CLASS[prompt]
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                preds.append((cls, (x1, y1, x2, y2)))

        gts = load_gt(lbl_dir / f"{img_path.stem}.txt", img_w, img_h)
        stats = match(preds, gts)
        for cls in (0, 1):
            for k in ("tp", "fp", "fn"):
                totals[cls][k] += stats[cls][k]

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample)} images processed")

    elapsed = time.time() - start

    report = {"per_class": {}, "elapsed_sec": round(elapsed, 2), "avg_fps": round(len(sample) / elapsed, 2)}
    names = {0: "worker", 1: "dangerous_vehicle"}
    for cls in (0, 1):
        tp, fp, fn = totals[cls]["tp"], totals[cls]["fp"], totals[cls]["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        report["per_class"][names[cls]] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
