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

CLASS_NAMES = {0: "worker", 1: "vehicle"}
BOX_COLORS = {0: (80, 220, 60), 1: (0, 165, 255)}  # BGR: green workers, orange vehicles
ALERT_BG = (30, 30, 220)  # red
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _draw_alert(frame, text):
    """Red alert banner pinned to the top-right corner."""
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    x2 = frame.shape[1] - 12
    x1 = x2 - tw - 20
    y1, y2 = 12, 12 + th + 16
    cv2.rectangle(frame, (x1, y1), (x2, y2), ALERT_BG, -1)
    cv2.putText(frame, text, (x1 + 10, y2 - 10), FONT, 0.7, (255, 255, 255), 2)


def _draw_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.xyxy]
        color = BOX_COLORS.get(det.cls, (200, 200, 200))
        label = f"{CLASS_NAMES.get(det.cls, det.cls)} {det.conf:.2f}"
        if det.flags:
            label += " " + " ".join(det.flags)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)), FONT, 0.5, color, 2)


def _draw_frame(frame, detections, hazards):
    _draw_boxes(frame, detections)
    if hazards:
        _draw_alert(frame, f"! POSSIBLE ACCIDENT x{len(hazards)}")
    return frame


def _draw_summary(frame, summary):
    """Overlay for summary detectors (Plan D/E): grounded boxes when present,
    plus counts line and top-right alert."""
    _draw_boxes(frame, summary.get("detections", []))
    line = f"workers={summary['workers']} vehicles={summary['vehicles']} ({summary['note']})"
    cv2.putText(frame, line, (12, 30), FONT, 0.6, (255, 255, 255), 2)
    if summary["hazard"]:
        _draw_alert(frame, "! POSSIBLE ACCIDENT")
    return frame


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def _reencode_h264(path: str):
    """OpenCV writes mp4v, which browsers cannot play. Re-encode to H.264 so
    st.video works. Leaves the original untouched if ffmpeg is unavailable."""
    import subprocess

    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return

    tmp = str(Path(path).with_suffix(".h264.mp4"))
    result = subprocess.run(
        [ffmpeg, "-y", "-i", path, "-c:v", "libx264", "-preset", "fast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", tmp],
        capture_output=True,
    )
    if result.returncode == 0 and Path(tmp).exists():
        Path(path).unlink()
        Path(tmp).rename(path)


def process_video(
    video_path: str,
    model_path: str,
    output_path: str,
    conf: float = 0.4,
    imgsz: int = 640,
    device: str = "0",
    frame_skip: int = 2,
    progress_cb=None,
    model: YOLO | None = None,
    detector=None,
    preview_cb=None,
    preview_every: int = 10,
):
    """
    Run detection over a video, write an annotated copy, and return a report dict.
    frame_skip: run the model every (frame_skip + 1) frames, reuse boxes on skipped
    frames to cut inference cost roughly in half for frame_skip=1, etc.
    Pass `detector` (any detectors.BaseDetector, e.g. cached across calls) to choose
    the approach; defaults to Plan A built from model_path / a preloaded `model`.
    """
    if detector is None:
        from detectors import PlanADetector
        detector = PlanADetector(weights_path=model_path, device=device, model=model).load()

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

    supports_boxes = getattr(detector, "supports_boxes", True)
    hazard_events = []
    class_counts = {0: 0, 1: 0}
    peak_counts = {0: 0, 1: 0}
    last_detections: list[Detection] = []
    last_summary = {"workers": 0, "vehicles": 0, "hazard": False, "note": "starting"}
    start = time.time()

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        run_model = frame_skip <= 0 or frame_idx % (frame_skip + 1) == 0

        if supports_boxes:
            if run_model:
                last_detections = detector.predict_frame(frame, conf=conf, imgsz=imgsz)
                frame_counts = {0: 0, 1: 0}
                for d in last_detections:
                    class_counts[d.cls] = class_counts.get(d.cls, 0) + 1
                    frame_counts[d.cls] = frame_counts.get(d.cls, 0) + 1
                for c in (0, 1):
                    peak_counts[c] = max(peak_counts[c], frame_counts.get(c, 0))

            hazards = find_proximity_hazards(last_detections, frame_diag)
            if hazards and run_model:
                hazard_events.append({
                    "frame": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 2),
                    "pairs": len(hazards),
                })
            annotated = _draw_frame(frame, last_detections, hazards)
        else:
            if run_model:
                last_summary = detector.summarize_frame(frame, conf=conf, imgsz=imgsz)
                class_counts[0] += last_summary["workers"]
                class_counts[1] += last_summary["vehicles"]
                peak_counts[0] = max(peak_counts[0], last_summary["workers"])
                peak_counts[1] = max(peak_counts[1], last_summary["vehicles"])
                if last_summary["hazard"]:
                    hazard_events.append({
                        "frame": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 2),
                        "pairs": 1,
                    })
            annotated = _draw_summary(frame, last_summary)

        writer.write(annotated)

        if preview_cb and frame_idx % preview_every == 0:
            preview_cb(annotated, frame_idx)

        frame_idx += 1
        if progress_cb and total_frames:
            progress_cb(frame_idx / total_frames)

    cap.release()
    writer.release()
    elapsed = time.time() - start
    _reencode_h264(output_path)

    report = {
        "video": str(Path(video_path).name),
        "approach": getattr(detector, "label", "Plan A — Fine-tuned YOLOv8"),
        "frames_processed": frame_idx,
        "duration_sec": round(elapsed, 2),
        "avg_fps": round(frame_idx / elapsed, 2) if elapsed > 0 else 0,
        "detections": {
            "worker": class_counts.get(0, 0),
            "dangerous_vehicle": class_counts.get(1, 0),
        },
        "peak": {
            "worker": peak_counts.get(0, 0),
            "dangerous_vehicle": peak_counts.get(1, 0),
        },
        "proximity_hazard_events": hazard_events,
        "total_hazard_events": len(hazard_events),
    }
    return report
