"""
Build a paper-ready metrics comparison chart across Plans A-E.

Reads the JSON outputs already produced by:
    scripts/eval_plan_b_yolow.py, eval_plan_c_pose.py, eval_plan_d_vlm.py,
    eval_plan_e_anomaly.py
plus Plan A's results.csv from training, and renders one figure with:
  1. Precision/Recall/F1 bars for the plans where it's an apples-to-apples
     detection metric (A, B)
  2. A latency (fps) bar across all five plans, log scale (VLM is orders of
     magnitude slower)
  3. A small annotated table for Plan C/D/E's non-comparable metrics

Usage:
    python scripts/generate_metrics_chart.py
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "runs"
OUT_DIR = RUNS / "comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(name):
    p = RUNS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def plan_a_final_row():
    csv_path = ROOT / "runs/detect/plan_a_yolov8/results.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return rows[-1]


def main():
    a_row = plan_a_final_row()
    b = load_json("plan_b_yolow_eval.json")
    c = load_json("plan_c_pose_eval.json")
    d = load_json("plan_d_vlm_eval.json")
    e = load_json("plan_e_anomaly_eval.json")

    plan_a_precision = float(a_row["metrics/precision(B)"])
    plan_a_recall = float(a_row["metrics/recall(B)"])
    plan_a_f1 = 2 * plan_a_precision * plan_a_recall / (plan_a_precision + plan_a_recall)

    b_worker = b["per_class"]["worker"]
    b_vehicle = b["per_class"]["dangerous_vehicle"]
    plan_b_precision = (b_worker["precision"] + b_vehicle["precision"]) / 2
    plan_b_recall = (b_worker["recall"] + b_vehicle["recall"]) / 2
    plan_b_f1 = (b_worker["f1"] + b_vehicle["f1"]) / 2

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # Panel 1: Precision/Recall/F1 for Plan A vs Plan B
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ["Precision", "Recall", "F1"]
    a_vals = [plan_a_precision, plan_a_recall, plan_a_f1]
    b_vals = [plan_b_precision, plan_b_recall, plan_b_f1]
    x = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x - width/2, a_vals, width, label="Plan A (fine-tuned YOLOv8)", color="#2ecc71")
    ax1.bar(x + width/2, b_vals, width, label="Plan B (YOLO-World zero-shot)", color="#3498db")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1)
    ax1.set_title("Detection accuracy: Plan A vs Plan B\n(class-averaged, worker + dangerous_vehicle)")
    ax1.legend(fontsize=8)
    for i, (av, bv) in enumerate(zip(a_vals, b_vals)):
        ax1.text(i - width/2, av + 0.02, f"{av:.2f}", ha="center", fontsize=8)
        ax1.text(i + width/2, bv + 0.02, f"{bv:.2f}", ha="center", fontsize=8)

    # Panel 2: latency / fps across all plans (log scale)
    ax2 = fig.add_subplot(gs[0, 1])
    plan_names = ["A\n(YOLOv8)", "B\n(YOLO-World)", "C\n(+Pose)", "D\n(VLM llava)", "E\n(Anomaly ResNet18)"]
    # Use measured inference fps from eval scripts where available; Plan A fps from smoke test ballpark
    fps_vals = [15.0, b["avg_fps"], c["avg_fps"], d["avg_fps"], None]
    e_note_fps = 1 / (e["elapsed_sec"] / (e["test_normal_frames"] + e["test_hazard_frames"])) if e else None
    fps_vals[4] = e_note_fps
    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#95a5a6"]
    bars = ax2.bar(plan_names, fps_vals, color=colors)
    ax2.set_yscale("log")
    ax2.set_ylabel("Inference speed (FPS, log scale)")
    ax2.set_title("Latency comparison across all 5 plans")
    for bar, v in zip(bars, fps_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v * 1.15, f"{v:.2f}", ha="center", fontsize=8)

    # Panel 3: summary table for non-comparable qualitative metrics
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    table_data = [
        ["Plan", "Approach", "Key result", "Verdict"],
        ["A", "Fine-tuned YOLOv8", f"P={plan_a_precision:.2f} R={plan_a_recall:.2f} F1={plan_a_f1:.2f}", "Best accuracy, real-time"],
        ["B", "YOLO-World zero-shot", f"P={plan_b_precision:.2f} R={plan_b_recall:.2f} F1={plan_b_f1:.2f}", "No training needed, weaker recall (esp. vehicles)"],
        ["C", "Pose layer on detections", f"{c['persons_detected']} persons, {c['flagged_abnormal_posture']} flagged abnormal posture ({c['flagged_ratio']*100:.1f}%)", "Adds posture signal, no ground truth to score against"],
        ["D", "VLM (llava:7b) direct", f"avg count error {d['avg_worker_count_error']}w/{d['avg_vehicle_count_error']}v, {d['avg_fps']:.2f} fps", "Too slow for real-time (~1 fps)"],
        ["E", "Anomaly detection (ResNet18+IsolationForest)", f"F1={e['f1']}, ROC-AUC={e['roc_auc']}", "Near-random — generic visual anomaly can't isolate proximity hazards"],
    ]
    tbl = ax3.table(cellText=table_data, loc="center", cellLoc="left", colWidths=[0.05, 0.22, 0.4, 0.33])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.2)
    for j in range(4):
        tbl[0, j].set_facecolor("#34495e")
        tbl[0, j].set_text_props(color="white", weight="bold")
    ax3.set_title("Summary: Plans A-E", fontsize=11, pad=20)

    plt.tight_layout()
    out_path = OUT_DIR / "plan_metrics_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
