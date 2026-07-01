"""
Plan E — Anomaly detection trained on "normal" site footage only.

We don't have a labeled anomaly dataset, so this uses our own test set as a
proxy: images with no proximity hazard (per Plan A's hazard_logic) are treated
as "normal", images with a proximity hazard are treated as the anomalous class
we want an unsupervised model to catch. A pretrained ResNet18 extracts a
global feature vector per frame; IsolationForest is fit on normal-only
features, then scored against a held-out mix of normal + hazard frames.

This tells us whether pure visual anomaly detection (no notion of "worker" or
"vehicle" at all) can distinguish hazard frames from calm ones.

Usage:
    python scripts/eval_plan_e_anomaly.py --n-train 300 --n-test 200
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from torchvision import models, transforms
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
from hazard_logic import Detection, find_proximity_hazards
import math


PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def label_hazard(detector, img_path: Path) -> bool:
    result = detector.predict(str(img_path), imgsz=640, conf=0.4, device="0", verbose=False)[0]
    if result.boxes is None:
        return False
    dets = []
    for box in result.boxes:
        cls = int(box.cls.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        dets.append(Detection(cls=cls, conf=float(box.conf.item()), xyxy=(x1, y1, x2, y2)))
    h, w = result.orig_shape
    diag = math.hypot(w, h)
    hazards = find_proximity_hazards(dets, diag)
    return len(hazards) > 0


def extract_features(model, device, img_paths):
    feats = []
    with torch.no_grad():
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            x = PREPROCESS(img).unsqueeze(0).to(device)
            f = model(x).squeeze().cpu().numpy()
            feats.append(f)
    return np.stack(feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="data/processed/images/test")
    parser.add_argument("--n-train", type=int, default=300, help="normal frames used to fit IsolationForest")
    parser.add_argument("--n-test", type=int, default=200, help="frames (mixed normal/hazard) used for scoring")
    parser.add_argument("--weights", default="runs/detect/plan_a_yolov8/weights/best.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="runs/plan_e_anomaly_eval.json")
    args = parser.parse_args()

    img_dir = Path(args.images)
    all_images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    random.seed(args.seed)
    random.shuffle(all_images)

    detector = YOLO(args.weights)

    print("Labeling frames as normal/hazard using Plan A detections...")
    normal, hazard = [], []
    pool = all_images[: args.n_train + args.n_test * 2]
    for i, p in enumerate(pool):
        if label_hazard(detector, p):
            hazard.append(p)
        else:
            normal.append(p)
        if len(normal) >= args.n_train + args.n_test // 2 and len(hazard) >= args.n_test // 2:
            break
        if (i + 1) % 100 == 0:
            print(f"  scanned {i+1}, normal={len(normal)}, hazard={len(hazard)}")

    train_normal = normal[: args.n_train]
    test_normal = normal[args.n_train: args.n_train + args.n_test // 2]
    test_hazard = hazard[: args.n_test // 2]
    print(f"train_normal={len(train_normal)}, test_normal={len(test_normal)}, test_hazard={len(test_hazard)}")

    if len(test_hazard) < 5:
        print("Not enough hazard frames found in this pool to evaluate meaningfully.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet.eval().to(device)

    start = time.time()
    train_feats = extract_features(resnet, device, train_normal)
    clf = IsolationForest(random_state=args.seed, contamination=0.1)
    clf.fit(train_feats)

    test_paths = test_normal + test_hazard
    test_labels = [0] * len(test_normal) + [1] * len(test_hazard)  # 1 = hazard (anomaly)
    test_feats = extract_features(resnet, device, test_paths)
    elapsed = time.time() - start

    scores = -clf.score_samples(test_feats)  # higher = more anomalous
    preds = clf.predict(test_feats)  # -1 = anomaly, 1 = normal
    pred_anomaly = [1 if p == -1 else 0 for p in preds]

    tp = sum(1 for t, p in zip(test_labels, pred_anomaly) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(test_labels, pred_anomaly) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(test_labels, pred_anomaly) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    try:
        auc = roc_auc_score(test_labels, scores) if len(set(test_labels)) > 1 else None
    except ValueError:
        auc = None

    report = {
        "train_normal_frames": len(train_normal),
        "test_normal_frames": len(test_normal),
        "test_hazard_frames": len(test_hazard),
        "precision_on_hazard_as_anomaly": round(precision, 3),
        "recall_on_hazard_as_anomaly": round(recall, 3),
        "f1": round(f1, 3),
        "roc_auc": round(auc, 3) if auc else None,
        "elapsed_sec": round(elapsed, 2),
        "note": "Proxy eval: hazard frames from Plan A used as stand-in for 'anomalous' since no true anomaly-labeled dataset exists.",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
