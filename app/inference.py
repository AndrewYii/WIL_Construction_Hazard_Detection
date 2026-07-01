"""
Video inference pipeline: detect worker / dangerous_vehicle, flag proximity
hazards, draw annotated frames, and produce a structured report.

Tuned for low latency:
  - half precision on GPU
  - frame skipping with box hold-over between detection passes
  - small imgsz by default (adjust for accuracy/speed tradeoff)
"""

import math
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from hazard_logic import Detection, find_proximity_hazards

CLASS_NAMES = {0: "worker", 1: "dangerous_vehicle"}
BOX_COLORS = {0: (60, 200, 60), 1: (40, 40, 230)}
HAZARD_COLOR = (0, 165, 255)


def _to_detections(result) -> list[Detection]:
    dets = []
    if result.boxes is None:
        return dets
    for box in result.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        dets.append(Detection(cls=cls, conf=conf, xyxy=(x1, y1, x2, y2)))
    return dets


def _draw_frame(frame, detections, hazards):
    hazard_workers = {id(w) for w, _ in hazards}
    hazard_vehicles = {id(v) for _, v in hazards}

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.xyxy]
        is_hazard = id(det) in hazard_workers or id(det) in hazard_vehicles
        color = HAZARD_COLOR if is_hazard else BOX_COLORS.get(det.cls, (200, 200, 200))
        label = f"{CLASS_NAMES.get(det.cls, det.cls)} {det.conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if hazards:
        cv2.putText(frame, f"PROXIMITY HAZARD x{len(hazards)}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, HAZARD_COLOR, 2)
    return frame


def process_video(
    video_path: str,
    model_path: str,
    output_path: str,
    conf: float = 0.4,
    imgsz: int = 640,
    device: str = "0",
    frame_skip: int = 2,
    progress_cb=None,
):
    """
    Run detection over a video, write an annotated copy, and return a report dict.
    frame_skip: run the model every (frame_skip + 1) frames, reuse boxes on skipped
    frames to cut inference cost roughly in half for frame_skip=1, etc.
    """
    model = YOLO(model_path)
    quantize = "fp16" if device != "cpu" else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_diag = math.hypot(width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    hazard_events = []
    class_counts = {0: 0, 1: 0}
    last_detections: list[Detection] = []
    start = time.time()

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        run_model = frame_skip <= 0 or frame_idx % (frame_skip + 1) == 0
        if run_model:
            result = model.predict(
                frame, imgsz=imgsz, conf=conf, device=device, quantize=quantize, verbose=False
            )[0]
            last_detections = _to_detections(result)
            for d in last_detections:
                class_counts[d.cls] = class_counts.get(d.cls, 0) + 1

        hazards = find_proximity_hazards(last_detections, frame_diag)
        if hazards and run_model:
            hazard_events.append({
                "frame": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 2),
                "pairs": len(hazards),
            })

        annotated = _draw_frame(frame, last_detections, hazards)
        writer.write(annotated)

        frame_idx += 1
        if progress_cb and total_frames:
            progress_cb(frame_idx / total_frames)

    cap.release()
    writer.release()
    elapsed = time.time() - start

    report = {
        "video": str(Path(video_path).name),
        "frames_processed": frame_idx,
        "duration_sec": round(elapsed, 2),
        "avg_fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0,
        "detections": {
            "worker": class_counts.get(0, 0),
            "dangerous_vehicle": class_counts.get(1, 0),
        },
        "proximity_hazard_events": hazard_events,
        "total_hazard_events": len(hazard_events),
    }
    return report
