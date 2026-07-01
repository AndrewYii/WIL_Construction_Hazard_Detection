"""
Build a paper-ready side-by-side comparison figure across Plans A-E, run on
the same sample test images.

Row = one test image. Columns = Plan A (trained YOLOv8), Plan B (YOLO-World
zero-shot), Plan C (pose overlay), Plan D (VLM text output), Plan E (anomaly
score). Saves a single PNG grid plus a separate metrics bar chart PNG.

Usage:
    python scripts/generate_comparison_figure.py
"""

import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ollama
import torch
from PIL import Image
from sklearn.ensemble import IsolationForest
from torchvision import models, transforms
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "app"))
from hazard_logic import Detection, find_proximity_hazards

IMG_DIR = ROOT / "data" / "processed" / "images" / "test"
LBL_DIR = ROOT / "data" / "processed" / "labels" / "test"
OUT_DIR = ROOT / "runs" / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_STEMS = [
    "-13-_png.rf.c1d75c2efe355241a72f3c0e873e68e0",
    "-1670-_png_jpg.rf.2ac42681fa66c65ac7594b891b529f77",
    "-2024-05-11-15-54-23_png.rf.961ab1d9e8a20bc2d0dcd88fce05b24f",
]

CLASS_NAMES = {0: "worker", 1: "vehicle"}
CLASS_COLORS = {0: "#2ecc71", 1: "#e74c3c"}

VLM_PROMPT = """You are inspecting a construction site photo for safety hazards.
Count the number of construction workers (people) and the number of heavy/dangerous
vehicles or machinery (trucks, excavators, cranes, bulldozers, forklifts) visible.
Then say if any worker appears dangerously close to a vehicle (proximity hazard).
Respond ONLY with compact JSON in this exact shape, no extra text:
{"workers": <int>, "vehicles": <int>, "proximity_hazard": <true/false>}"""


def find_image(stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = IMG_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


def draw_boxes(ax, img, detections, title):
    ax.imshow(img)
    for cls, (x1, y1, x2, y2), conf in detections:
        color = CLASS_COLORS.get(cls, "yellow")
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        ax.text(x1, max(y1 - 4, 0), f"{CLASS_NAMES.get(cls,'?')} {conf:.2f}",
                color="white", fontsize=6, backgroundcolor=color)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def draw_pose(ax, img, result, title):
    ax.imshow(img)
    skeleton = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    if result.keypoints is not None:
        for kpts in result.keypoints.data:
            kpts = kpts.tolist()
            for a, b in skeleton:
                if kpts[a][2] > 0.3 and kpts[b][2] > 0.3:
                    ax.plot([kpts[a][0], kpts[b][0]], [kpts[a][1], kpts[b][1]], color="#3498db", linewidth=1.5)
            for x, y, c in kpts:
                if c > 0.3:
                    ax.plot(x, y, "o", color="#f39c12", markersize=2)
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#3498db", linewidth=1))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def draw_text_panel(ax, img, text, title):
    ax.imshow(img)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=8, color="white",
            va="top", ha="left", backgroundcolor="black", wrap=True)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def parse_vlm(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"workers": "?", "vehicles": "?", "proximity_hazard": "?"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"workers": "?", "vehicles": "?", "proximity_hazard": "?"}


def main():
    print("Loading models...")
    plan_a = YOLO(str(ROOT / "runs/detect/plan_a_yolov8/weights/best.pt"))
    plan_b = YOLO("yolov8s-world.pt")
    plan_b.set_classes(["person", "worker", "construction worker", "truck", "excavator", "crane", "vehicle"])
    plan_b_prompt_to_class = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}
    plan_c = YOLO("yolov8s-pose.pt")

    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Fitting anomaly model on normal frames (Plan E)...")
    all_imgs = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))
    normal_sample = [p for p in all_imgs[:150] if p.stem not in SAMPLE_STEMS][:80]
    with torch.no_grad():
        train_feats = np.stack([
            resnet(preprocess(Image.open(p).convert("RGB")).unsqueeze(0)).squeeze().numpy()
            for p in normal_sample
        ])
    iso = IsolationForest(random_state=42, contamination=0.1)
    iso.fit(train_feats)

    n_rows = len(SAMPLE_STEMS)
    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
    col_titles = ["Plan A (YOLOv8 fine-tuned)", "Plan B (YOLO-World zero-shot)",
                  "Plan C (Pose overlay)", "Plan D (VLM: llava)", "Plan E (Anomaly score)"]

    for row, stem in enumerate(SAMPLE_STEMS):
        img_path = find_image(stem)
        pil_img = Image.open(img_path).convert("RGB")

        # Plan A
        r = plan_a.predict(str(img_path), imgsz=640, conf=0.4, device="0", verbose=False)[0]
        dets_a = [(int(b.cls.item()), tuple(float(v) for v in b.xyxy[0]), float(b.conf.item())) for b in (r.boxes or [])]
        draw_boxes(axes[row][0], pil_img, dets_a, col_titles[0] if row == 0 else "")

        # Plan B
        r = plan_b.predict(str(img_path), imgsz=640, conf=0.15, device="0", verbose=False)[0]
        dets_b = []
        for b in (r.boxes or []):
            prompt_idx = int(b.cls.item())
            cls = plan_b_prompt_to_class.get(prompt_idx, 0)
            dets_b.append((cls, tuple(float(v) for v in b.xyxy[0]), float(b.conf.item())))
        draw_boxes(axes[row][1], pil_img, dets_b, col_titles[1] if row == 0 else "")

        # Plan C
        r = plan_c.predict(str(img_path), imgsz=640, conf=0.4, device="0", verbose=False)[0]
        draw_pose(axes[row][2], pil_img, r, col_titles[2] if row == 0 else "")

        # Plan D
        response = ollama.chat(model="llava:7b", messages=[{"role": "user", "content": VLM_PROMPT, "images": [str(img_path)]}])
        parsed = parse_vlm(response["message"]["content"])
        text = f"workers: {parsed['workers']}\nvehicles: {parsed['vehicles']}\nproximity_hazard: {parsed['proximity_hazard']}"
        draw_text_panel(axes[row][3], pil_img, text, col_titles[3] if row == 0 else "")

        # Plan E
        feat = resnet(preprocess(pil_img).unsqueeze(0)).detach().numpy()
        score = -iso.score_samples(feat)[0]
        is_anomaly = iso.predict(feat)[0] == -1
        text = f"anomaly score: {score:.3f}\nflagged: {is_anomaly}"
        draw_text_panel(axes[row][4], pil_img, text, col_titles[4] if row == 0 else "")

        print(f"  row {row+1}/{n_rows} done ({stem})")

    plt.tight_layout()
    out_path = OUT_DIR / "plan_comparison_grid.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
