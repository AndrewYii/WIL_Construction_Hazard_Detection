"""
Plan C — Pose estimation layered on top of worker detections.

No public dataset labels worker posture/falls, so this is a qualitative eval:
run YOLOv8-pose on the same test images, derive a simple "fallen/abnormal
posture" heuristic from keypoints (bounding box aspect ratio + shoulder-hip
verticality), and report how often it fires plus latency versus Plan A/B.

Usage:
    python scripts/eval_plan_c_pose.py --n-samples 200
"""

import argparse
import json
import random
import time
from pathlib import Path

from ultralytics import YOLO

# COCO keypoint indices used by YOLOv8-pose
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


def posture_flag(keypoints, box):
    """Very rough heuristic: standing workers are taller than wide, and
    shoulders sit clearly above hips. Flag anything that looks horizontal."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if h <= 0:
        return False
    aspect = w / h
    if aspect > 1.2:
        return True  # wider than tall -> likely lying down

    kpts = keypoints  # shape (17, 3): x, y, conf
    shoulder_y = [kpts[i][1] for i in (L_SHOULDER, R_SHOULDER) if kpts[i][2] > 0.3]
    hip_y = [kpts[i][1] for i in (L_HIP, R_HIP) if kpts[i][2] > 0.3]
    if shoulder_y and hip_y:
        avg_shoulder = sum(shoulder_y) / len(shoulder_y)
        avg_hip = sum(hip_y) / len(hip_y)
        if avg_shoulder >= avg_hip:  # shoulders not above hips -> abnormal
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/processed/images/test")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/plan_c_pose_eval.json")
    args = parser.parse_args()

    img_dir = Path(args.images)
    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    random.seed(args.seed)
    sample = random.sample(all_images, min(args.n_samples, len(all_images)))
    print(f"Evaluating on {len(sample)} / {len(all_images)} test images")

    model = YOLO("yolov8s-pose.pt")

    total_persons = 0
    flagged_abnormal = 0
    start = time.time()

    for i, img_path in enumerate(sample):
        result = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        if result.keypoints is None or result.boxes is None:
            continue
        for box, kpts in zip(result.boxes, result.keypoints.data):
            total_persons += 1
            box_xyxy = [float(v) for v in box.xyxy[0]]
            kpts_list = kpts.tolist()
            if posture_flag(kpts_list, box_xyxy):
                flagged_abnormal += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample)} images processed")

    elapsed = time.time() - start

    report = {
        "images_evaluated": len(sample),
        "persons_detected": total_persons,
        "flagged_abnormal_posture": flagged_abnormal,
        "flagged_ratio": round(flagged_abnormal / total_persons, 3) if total_persons else 0,
        "elapsed_sec": round(elapsed, 2),
        "avg_fps": round(len(sample) / elapsed, 2),
        "note": "Heuristic only — no ground truth for posture/fall exists in current datasets.",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
