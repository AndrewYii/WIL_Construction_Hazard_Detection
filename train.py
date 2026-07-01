"""
Plan A — Fine-tune YOLOv8 on merged construction site dataset.
Classes: worker (0), dangerous_vehicle (1)

Usage:
    python train.py
    python train.py --model yolov8m.pt --epochs 100 --batch 16
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8s.pt", help="Pretrained weights (n/s/m/l/x)")
    parser.add_argument("--data", default="data/data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="GPU device id, or 'cpu'")
    parser.add_argument("--name", default="plan_a_yolov8", help="Run name under runs/detect/")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    project_root = Path(__file__).parent
    data_yaml = project_root / args.data
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found at {data_yaml}")

    runs_dir = project_root / "runs" / "detect"

    model = YOLO(args.model)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(runs_dir),
        name=args.name,
        workers=args.workers,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
    )

    best = runs_dir / args.name / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights : {best}")
    print(f"Results saved: {runs_dir / args.name}")


if __name__ == "__main__":
    main()
