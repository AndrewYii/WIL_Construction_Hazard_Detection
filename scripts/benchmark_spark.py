"""
FPS benchmark for the DGX Spark (or any machine): times YOLO variants on one
construction image and prints a markdown table for the project report.

    python scripts/benchmark_spark.py [--image path.jpg] [--weights best.pt]
        [--imgsz 640] [--append-readme]

- yolov8n/s/m/l/x and yolo11n/s/m auto-download on first run
- 10 warmup + 100 timed inferences each, half precision on GPU
- verdict line: largest model sustaining >= 30 FPS
- --append-readme appends the table to README.md under
  "Performance on DGX Spark"
"""

import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
          "yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]

WARMUP = 10
TIMED = 100


def find_default_image() -> np.ndarray:
    import cv2
    for pattern in ("testdata/**/*.jpg", "data/**/*.jpg", "data/**/*.png"):
        for p in ROOT.glob(pattern):
            img = cv2.imread(str(p))
            if img is not None:
                print(f"[bench] using image: {p}")
                return img
    print("[bench] no test image found in repo — using synthetic 1280x720 noise")
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)


def bench_one(model_name: str, image, imgsz: int, device: str) -> dict | None:
    import torch
    from ultralytics import YOLO

    try:
        model = YOLO(model_name)
    except Exception as exc:
        print(f"[bench] skip {model_name}: {exc}")
        return None

    half = device != "cpu" and torch.cuda.is_available()
    kwargs = dict(imgsz=imgsz, device=device, half=half, verbose=False)

    for _ in range(WARMUP):
        model.predict(image, **kwargs)
    if half:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(TIMED):
        model.predict(image, **kwargs)
    if half:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
    ms = elapsed / TIMED * 1000
    return {"model": Path(model_name).stem, "params_m": n_params,
            "fps": 1000 / ms, "ms": ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="Construction image to benchmark on")
    ap.add_argument("--weights", default=str(ROOT / "runs/detect/plan_a_yolov8/weights/best.pt"),
                    help="Also benchmark the project's fine-tuned weights")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--append-readme", action="store_true")
    args = ap.parse_args()

    import cv2
    import torch
    if args.device != "cpu" and not torch.cuda.is_available():
        print("[bench] CUDA not available — falling back to CPU (numbers will be low)")
        args.device = "cpu"

    image = cv2.imread(args.image) if args.image else find_default_image()
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    targets = list(MODELS)
    if args.weights and Path(args.weights).exists():
        targets.append(args.weights)
    else:
        print(f"[bench] fine-tuned weights not found ({args.weights}) — skipping")

    rows = []
    for name in targets:
        label = "best.pt (fine-tuned)" if name == args.weights else name
        print(f"[bench] {label} ...")
        r = bench_one(name, image, args.imgsz, args.device)
        if r:
            if name == args.weights:
                r["model"] = "best.pt (fine-tuned)"
            rows.append(r)
            print(f"        {r['fps']:.1f} FPS ({r['ms']:.1f} ms)")

    if not rows:
        raise SystemExit("No models benchmarked.")

    lines = ["| model | params (M) | FPS | ms/frame |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['params_m']:.1f} | {r['fps']:.1f} | {r['ms']:.1f} |")
    realtime = [r for r in rows if r["fps"] >= 30]
    if realtime:
        biggest = max(realtime, key=lambda r: r["params_m"])
        verdict = (f"**Verdict:** largest model sustaining >= 30 FPS at imgsz "
                   f"{args.imgsz}: `{biggest['model']}` "
                   f"({biggest['fps']:.1f} FPS, {biggest['params_m']:.1f}M params).")
    else:
        verdict = "**Verdict:** no benchmarked model sustained 30 FPS on this device."
    table = "\n".join(lines) + "\n\n" + verdict + "\n"

    print("\n" + table)

    if args.append_readme:
        readme = ROOT / "README.md"
        stamp = time.strftime("%Y-%m-%d")
        section = (f"\n## Performance on DGX Spark\n\n"
                   f"Benchmarked {stamp}, imgsz {args.imgsz}, "
                   f"{WARMUP} warmup + {TIMED} timed inferences, "
                   f"{'FP16' if args.device != 'cpu' else 'CPU FP32'}.\n\n{table}")
        with open(readme, "a", encoding="utf-8") as f:
            f.write(section)
        print(f"[bench] appended to {readme}")


if __name__ == "__main__":
    main()
