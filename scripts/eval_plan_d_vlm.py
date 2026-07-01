"""
Plan D — Direct VLM hazard detection, no separate detector.

Sends each test image straight to a local Ollama vision model (llava:7b) and
asks it to report worker/vehicle counts and any hazards, in JSON. Compares
counts against our existing YOLO-format ground truth as a rough correctness
check, and measures latency (the real point of this comparison — VLMs are
expected to be much slower than a dedicated detector).

Requires: `ollama pull llava:7b` and the Ollama service running locally.

Usage:
    python scripts/eval_plan_d_vlm.py --n-samples 30
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import ollama

PROMPT = """You are inspecting a construction site photo for safety hazards.
Count the number of construction workers (people) and the number of heavy/dangerous
vehicles or machinery (trucks, excavators, cranes, bulldozers, forklifts) visible.
Then say if any worker appears dangerously close to a vehicle (proximity hazard).
Respond ONLY with compact JSON in this exact shape, no extra text:
{"workers": <int>, "vehicles": <int>, "proximity_hazard": <true/false>}"""


def load_gt_counts(label_path: Path):
    workers = vehicles = 0
    if not label_path.exists():
        return workers, vehicles
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(parts[0])
        if cls == 0:
            workers += 1
        elif cls == 1:
            vehicles += 1
    return workers, vehicles


def parse_response(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/processed/images/test")
    parser.add_argument("--labels", default="data/processed/labels/test")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--model", default="llava:7b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/plan_d_vlm_eval.json")
    args = parser.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    random.seed(args.seed)
    sample = random.sample(all_images, min(args.n_samples, len(all_images)))
    print(f"Evaluating on {len(sample)} / {len(all_images)} test images using {args.model}")

    results = []
    parse_failures = 0
    start = time.time()

    for i, img_path in enumerate(sample):
        gt_workers, gt_vehicles = load_gt_counts(lbl_dir / f"{img_path.stem}.txt")

        t0 = time.time()
        response = ollama.chat(
            model=args.model,
            messages=[{"role": "user", "content": PROMPT, "images": [str(img_path)]}],
        )
        latency = time.time() - t0

        parsed = parse_response(response["message"]["content"])
        if parsed is None:
            parse_failures += 1
            parsed = {"workers": None, "vehicles": None, "proximity_hazard": None}

        results.append({
            "image": img_path.name,
            "gt_workers": gt_workers,
            "gt_vehicles": gt_vehicles,
            "pred": parsed,
            "latency_sec": round(latency, 2),
        })
        print(f"  [{i+1}/{len(sample)}] {img_path.name}: gt=({gt_workers}w,{gt_vehicles}v) "
              f"pred={parsed} ({latency:.1f}s)")

    elapsed = time.time() - start

    worker_errs = [abs(r["gt_workers"] - r["pred"]["workers"]) for r in results if r["pred"]["workers"] is not None]
    vehicle_errs = [abs(r["gt_vehicles"] - r["pred"]["vehicles"]) for r in results if r["pred"]["vehicles"] is not None]

    report = {
        "model": args.model,
        "images_evaluated": len(sample),
        "parse_failures": parse_failures,
        "avg_worker_count_error": round(sum(worker_errs) / len(worker_errs), 2) if worker_errs else None,
        "avg_vehicle_count_error": round(sum(vehicle_errs) / len(vehicle_errs), 2) if vehicle_errs else None,
        "avg_latency_sec_per_image": round(elapsed / len(sample), 2),
        "avg_fps": round(len(sample) / elapsed, 4),
        "details": results,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps({k: v for k, v in report.items() if k != "details"}, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
