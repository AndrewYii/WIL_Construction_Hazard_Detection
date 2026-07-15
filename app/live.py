"""
Real-time construction hazard detection with on-the-spot audio alerts.

Pipeline (all threads, never blocking each other):
  capture thread  -> latest-frame slot (stale frames dropped)
  detection loop  -> YOLO -> hazard layer (proximity / vehicle / height)
                  -> alert engine (voice + siren + events.jsonl)
                  -> publishes annotated frame
  HTTP threads    -> JPEG-encode the latest annotated frame per client
                     (MJPEG), serve dashboard + JSON status + LLM report

Run on the Spark (headless, browser is the monitor):
    python app/live.py --weights runs/detect/plan_a_yolov8/weights/best.pt \
        --source 0 --headless --port 8090

Rehearsal with no GPU / no weights / no camera:
    python app/live.py --video clip.mp4 --mock --headless

Without --headless a cv2.imshow window opens instead (press q to quit).
Phase 2 reports come from the Ollama server on the DGX Spark; set
OLLAMA_HOST=http://<spark-ip>:11434 (see docs/SPARK_SETUP.md).
"""

import argparse
import json
import math
import queue
import random
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

import config
from alerts import AlertEngine, AudioPlayer
from hazard_logic import (Detection, VehicleMotionTracker, find_height_hazards,
                          find_proximity_hazards)

FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_COLORS = {0: (80, 220, 60), 1: (0, 165, 255)}  # BGR: workers green, vehicles orange
CLASS_NAMES = {0: "worker", 1: "vehicle"}
ALERT_BG = (30, 30, 220)

HAZARD_LABELS = {
    "proximity": "WORKER TOO CLOSE TO VEHICLE",
    "vehicle": "VEHICLE MOVING IN WORK ZONE",
    "height": "WORKER AT HEIGHT",
}


_CAM_CACHE = {"t": 0.0, "list": []}


def list_local_cameras(current=None, max_probe: int = 5) -> list[dict]:
    """Meet-style device list: probe OpenCV indices, pair with OS device
    names (Windows PnP / Linux v4l2). The index currently held open by the
    monitor can't be probed, so it is kept and marked active. Cached 60s."""
    import platform
    now = time.time()
    if now - _CAM_CACHE["t"] < 60 and _CAM_CACHE["list"]:
        return _CAM_CACHE["list"]
    names = []
    system = platform.system()
    try:
        if system == "Windows":
            import subprocess
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_PnPEntity | Where-Object "
                 "{$_.PNPClass -in 'Camera','Image'} | "
                 "Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=10).stdout
            names = [line.strip() for line in out.splitlines() if line.strip()]
        else:
            import glob
            for path in sorted(glob.glob("/sys/class/video4linux/video*/name")):
                with open(path) as f:
                    names.append(f.read().strip())
    except Exception:
        pass
    backend = cv2.CAP_DSHOW if system == "Windows" else cv2.CAP_ANY
    cams = []
    for i in range(max_probe):
        label = names[i] if i < len(names) else f"Camera {i}"
        if current is not None and i == current:
            cams.append({"index": i, "label": label})
            continue
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            cams.append({"index": i, "label": label})
        cap.release()
    _CAM_CACHE.update(t=now, list=cams)
    return cams


def camera_off_frame(shape=(360, 640, 3)):
    """Placeholder pushed to the streams when the camera is toggled off, so
    viewers see an explicit OFF state instead of a frozen last frame."""
    import numpy as np
    frame = np.zeros(shape, dtype=np.uint8)
    h, w = shape[:2]
    text = "CAMERA OFF"
    (tw, th), _ = cv2.getTextSize(text, FONT, 1.1, 2)
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2), FONT, 1.1,
                (120, 120, 120), 2)
    return frame


# --------------------------------------------------------------------------
# Frame plumbing
# --------------------------------------------------------------------------

class LatestFrame:
    """Single-slot frame holder: writers overwrite, readers take the newest.
    Guarantees the consumer never works on a stale backlog."""

    def __init__(self):
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self.closed = False

    def put(self, frame):
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()

    def get(self, last_seq: int, timeout: float = 1.0):
        """Block until a frame newer than last_seq exists (or timeout).
        Returns (frame, seq) — frame is None on timeout/close."""
        with self._cond:
            if self._seq <= last_seq and not self.closed:
                self._cond.wait(timeout)
            if self._seq > last_seq and self._frame is not None:
                return self._frame, self._seq
            return None, last_seq

    def close(self):
        with self._cond:
            self.closed = True
            self._cond.notify_all()


class CaptureThread(threading.Thread):
    """Reads the camera/RTSP/file as fast as it arrives into a LatestFrame.
    Files are looped endlessly so rehearsals can run for as long as needed."""

    def __init__(self, source, slot: LatestFrame, loop_file: bool):
        super().__init__(daemon=True)
        self.source = source
        self.slot = slot
        self.loop_file = loop_file
        self.stop_flag = threading.Event()
        self.fps_hint = 25.0
        # SourceManager clears this when hot-swapping cameras so a retiring
        # thread doesn't shut the shared slot under the new one
        self.close_on_exit = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[capture] ERROR: cannot open source {self.source!r}")
            if self.close_on_exit:
                self.slot.close()
            return
        self.fps_hint = cap.get(cv2.CAP_PROP_FPS) or 25.0
        delay = 1.0 / self.fps_hint if self.loop_file else 0.0
        while not self.stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                if self.loop_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            self.slot.put(frame)
            if delay:  # pace file playback at native FPS
                time.sleep(delay)
        cap.release()
        if self.close_on_exit:
            self.slot.close()


def parse_source(raw: str):
    return int(raw) if isinstance(raw, str) and raw.isdigit() else raw


def public_source(src) -> str:
    """What the HTTP API may reveal about the current source: camera index,
    URL host, or file basename — never a filesystem path or full URL."""
    s = str(src)
    if isinstance(src, int) or s.isdigit():
        return s
    if "://" in s:
        from urllib.parse import urlparse
        parsed = urlparse(s)
        return f"{parsed.scheme}://{parsed.hostname or ''}"
    return Path(s).name


class SourceManager:
    """Hot-swap the camera at runtime (webcam <-> RealSense <-> phone URL)
    without touching detection or connected dashboard clients."""

    def __init__(self, in_slot: LatestFrame):
        self.in_slot = in_slot
        self.capture: CaptureThread | None = None
        self.current = None
        self.camera_on = False
        self._loop_file = False
        self._lock = threading.Lock()

    def start(self, source, loop_file: bool = False, critical: bool = False) -> bool:
        """critical=True: failure to open ends the session (initial startup).
        Hot-swaps use critical=False so a missing camera (e.g. RealSense not
        plugged in yet) just stalls the stream until the user switches back."""
        with self._lock:
            if self.capture and self.capture.is_alive():
                self.capture.close_on_exit = False
                self.capture.stop_flag.set()
            cap = CaptureThread(source, self.in_slot, loop_file)
            cap.close_on_exit = critical
            cap.start()
            self.capture = cap
            self.current = source
            self.camera_on = True
            self._loop_file = loop_file
            return True

    def set_camera(self, on: bool):
        """Meeting-style camera toggle: off releases the device entirely
        (privacy — its LED goes dark), on reopens the remembered source."""
        if on:
            if not self.camera_on and self.current is not None:
                self.start(self.current, loop_file=self._loop_file)
            return
        with self._lock:
            if self.capture and self.capture.is_alive():
                self.capture.close_on_exit = False
                self.capture.stop_flag.set()
            self.camera_on = False

    def stop(self):
        with self._lock:
            if self.capture:
                self.capture.stop_flag.set()


# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------

class MockDetector:
    """Synthetic worker + excavator that wander and periodically meet, so the
    whole alert/dashboard/audio chain can be rehearsed with no GPU."""

    label = "Mock detector (rehearsal)"

    def __init__(self):
        self.t = 0.0

    def predict_frame(self, frame, conf, imgsz):
        h, w = frame.shape[:2]
        self.t += 0.04
        # worker orbits the center; vehicle sweeps left-right and meets the
        # worker roughly every ~12 seconds
        wx = w * (0.5 + 0.30 * math.sin(self.t))
        wy = h * (0.62 + 0.05 * math.sin(self.t * 2.3))
        vx = w * (0.5 + 0.42 * math.sin(self.t * 0.5))
        vy = h * 0.60
        jitter = lambda: random.uniform(-2, 2)
        worker = Detection(0, 0.93, (wx - 25 + jitter(), wy - 70, wx + 25, wy + 70))
        vehicle = Detection(1, 0.90, (vx - 110 + jitter(), vy - 80, vx + 110, vy + 80))
        return [worker, vehicle]


def build_detector(args):
    if args.mock:
        return MockDetector()
    from detectors import PlanADetector
    if not Path(args.weights).exists():
        raise SystemExit(
            f"Weights not found: {args.weights}\n"
            "Pass --weights <path to best.pt> or use --mock for rehearsal.")
    return PlanADetector(weights_path=args.weights, device=args.device).load()


# --------------------------------------------------------------------------
# Session state shared with the HTTP server
# --------------------------------------------------------------------------

class SessionState:
    def __init__(self, engine: AlertEngine):
        self.engine = engine
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.frames = 0
        self.fps = 0.0
        self.workers_now = 0
        self.vehicles_now = 0
        self.peak = {"worker": 0, "dangerous_vehicle": 0}
        self.totals = {"worker": 0, "dangerous_vehicle": 0}
        self.active_hazards: list[str] = []
        self.proximity_events: list[dict] = []
        self.detector_label = ""
        self.spark_status: dict = {"up": False, "host": config.OLLAMA_HOST, "models": []}
        self.incidents: list[dict] = []   # VLM scene notes per fired alert
        self.live_report: str = ""        # auto-refreshed LLM report
        self.viewers = 0                  # open MJPEG connections
        self.auto_off = True              # camera auto-off when last viewer leaves
        self.mirror = False               # horizontal flip (front-facing cameras)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "uptime_sec": round(time.time() - self.start_time, 1),
                "frames": self.frames,
                "fps": round(self.fps, 1),
                "workers": self.workers_now,
                "vehicles": self.vehicles_now,
                "peak": dict(self.peak),
                "active_hazards": list(self.active_hazards),
                "alert_counts": self.engine.counts(),
                "recent_alerts": self.engine.recent[-12:],
                "detector": self.detector_label,
                "spark": self.spark_status,
                "hazard_labels": HAZARD_LABELS,
                "incidents": self.incidents[-8:],
                "live_report": self.live_report,
                "viewers": self.viewers,
                "auto_off": self.auto_off,
                "mirror": self.mirror,
            }

    def to_report_dict(self) -> dict:
        with self.lock:
            elapsed = max(time.time() - self.start_time, 1e-3)
            return {
                "video": "live stream",
                "approach": self.detector_label,
                "frames_processed": self.frames,
                "duration_sec": round(elapsed, 2),
                "avg_fps": round(self.frames / elapsed, 2),
                "detections": dict(self.totals),
                "peak": dict(self.peak),
                "proximity_hazard_events": list(self.proximity_events),
                "total_hazard_events": len(self.proximity_events),
                "incident_notes": self.incidents[-12:],
            }


class IncidentAnalyst(threading.Thread):
    """On-the-spot reasoning path. The instant voice alert has already fired
    (milliseconds); this thread takes the flagged snapshot to the Spark:
      1. VLM describes the scene (who is at risk, which machinery, layout)
      2. the LLM rewrites the live safety report with those notes,
         throttled to one rewrite per REPORT_REFRESH_SEC
    Bounded queue, drops snapshots under backlog — detection never waits."""

    VLM_PROMPT = (
        "A {kind} hazard alert just fired on this construction site camera "
        "({message}). In 2-3 short factual sentences, describe the scene: "
        "who is at risk, what machinery is involved, and their spatial "
        "arrangement. No preamble, no speculation beyond the image."
    )

    def __init__(self, state: SessionState,
                 report_refresh_sec: float = config.REPORT_REFRESH_SEC):
        super().__init__(daemon=True)
        self.state = state
        self.report_refresh_sec = report_refresh_sec
        self._jobs: queue.Queue = queue.Queue(maxsize=4)
        self._last_report = 0.0

    def submit(self, event: dict, frame):
        try:
            self._jobs.put_nowait((event, frame))
        except queue.Full:
            pass  # analysis is best-effort; the alert itself already fired

    def run(self):
        from llm_client import get_client
        from report_generation import generate_hazard_report
        client = get_client()
        while True:
            event, frame = self._jobs.get()
            note = None
            try:
                ok, jpg = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok and client.is_up():
                    prompt = self.VLM_PROMPT.format(
                        kind=event.get("type", "safety"),
                        message=event.get("message", ""))
                    note = client.describe_image(prompt, jpg.tobytes())
            except Exception:
                note = None
            with self.state.lock:
                self.state.incidents.append({
                    "iso": event.get("iso", ""),
                    "type": event.get("type", ""),
                    "note": (note or "").strip()[:500] or "scene review unavailable",
                })
                del self.state.incidents[:-50]
            now = time.time()
            if now - self._last_report >= self.report_refresh_sec:
                self._last_report = now
                try:
                    text = generate_hazard_report(self.state.to_report_dict())
                    with self.state.lock:
                        self.state.live_report = text
                except Exception:
                    pass


def spark_status_poller(state: SessionState, interval: float = 30.0):
    """Background refresh of the Spark Ollama status for the dashboard.
    Also keeps the report/VLM models warm so alert-time reasoning is fast."""
    from llm_client import get_client
    client = get_client()
    warmed = 0.0
    while True:
        try:
            status = client.status()
        except Exception:
            status = {"up": False, "host": config.OLLAMA_HOST, "models": []}
        with state.lock:
            state.spark_status = status
        if status.get("up") and time.time() - warmed > 600:
            client.warm()  # re-pin models within their 30m keep_alive window
            warmed = time.time()
        time.sleep(interval)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def _draw_subtitle(frame, text):
    """Broadcast-style caption bar near the bottom of the frame — raised so
    the dashboard's floating camera controls never cover it, readable on a
    phone screen, and preserved in any recording of the stream."""
    h, w = frame.shape[:2]
    scale = max(0.55, min(w / 1100, 1.0))
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    x = max((w - tw) // 2, 8)
    y = h - max(int(h * 0.16), 60)
    cv2.rectangle(frame, (x - 14, y - th - 12), (x + tw + 14, y + 10), (20, 20, 190), -1)
    cv2.putText(frame, text, (x, y), FONT, scale, (255, 255, 255), 2)


def annotate(frame, detections, hazard_types, hazard_pairs):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.xyxy]
        color = BOX_COLORS.get(det.cls, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES.get(det.cls, det.cls)} {det.conf:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 14)), FONT, 0.5, color, 2)
    # red link line between each hazard pair
    for w, v in hazard_pairs:
        wx, wy = (int((w.xyxy[0] + w.xyxy[2]) / 2), int((w.xyxy[1] + w.xyxy[3]) / 2))
        vx, vy = (int((v.xyxy[0] + v.xyxy[2]) / 2), int((v.xyxy[1] + v.xyxy[3]) / 2))
        cv2.line(frame, (wx, wy), (vx, vy), (0, 0, 255), 2)
    y = 34
    for kind in hazard_types:
        text = f"! {HAZARD_LABELS.get(kind, kind.upper())}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.75, 2)
        x2 = frame.shape[1] - 12
        cv2.rectangle(frame, (x2 - tw - 20, y - th - 8), (x2, y + 8), ALERT_BG, -1)
        cv2.putText(frame, text, (x2 - tw - 10, y), FONT, 0.75, (255, 255, 255), 2)
        y += th + 24
    if hazard_types:
        _draw_subtitle(frame, "WARNING: " + "  +  ".join(
            HAZARD_LABELS.get(k, k.upper()) for k in hazard_types))
    return frame


def draw_hud(frame, state: SessionState):
    hud = f"{state.fps:.1f} FPS | workers {state.workers_now} | vehicles {state.vehicles_now}"
    cv2.putText(frame, hud, (12, 26), FONT, 0.65, (0, 0, 0), 4)
    cv2.putText(frame, hud, (12, 26), FONT, 0.65, (255, 255, 255), 2)
    return frame


def _side(x: float, frame_w: float) -> str:
    third = frame_w / 3
    return "left" if x < third else ("right" if x > 2 * third else "center")


def compose_alert_messages(hazard_pairs, moving_vehicles, elevated, workers,
                           frame_w: float) -> dict[str, str]:
    """Scene-specific spoken phrases built from detection geometry — composed
    in microseconds at the moment of detection, no model involved, so the
    descriptive voice alert still fires on the spot."""
    def at(det):
        side = _side((det.xyxy[0] + det.xyxy[2]) / 2, frame_w)
        return "in the center" if side == "center" else f"on the {side}"

    msgs = {}
    if hazard_pairs:
        n = len({id(w) for w, _ in hazard_pairs})
        who = "One worker" if n == 1 else f"{n} workers"
        msgs["proximity"] = (f"Warning! {who} too close to vehicle "
                             f"{at(hazard_pairs[0][1])}. Move away now.")
    if moving_vehicles:
        n_w = len(workers)
        tail = (f", {n_w} worker{'s' if n_w != 1 else ''} nearby." if n_w else ".")
        msgs["vehicle"] = f"Caution! Vehicle moving {at(moving_vehicles[0])}{tail}"
    if elevated:
        msgs["height"] = (f"Warning! Worker at height {at(elevated[0])}. "
                          "Check fall protection.")
    return msgs


# --------------------------------------------------------------------------
# Detection loop
# --------------------------------------------------------------------------

def detection_loop(args, detector, in_slot: LatestFrame,
                   out_slot: LatestFrame, state: SessionState, engine: AlertEngine,
                   stop_flag: threading.Event, analyst: IncidentAnalyst | None = None,
                   raw_slot: LatestFrame | None = None):
    motion = None
    frame_diag = None
    frame_shape = None
    last_seq = 0
    fps_smooth = None

    while not stop_flag.is_set():
        frame, last_seq = in_slot.get(last_seq)
        if frame is None:
            if in_slot.closed:
                break
            continue
        t0 = time.time()

        if frame.shape[:2] != frame_shape:  # first frame, or camera swapped
            frame_shape = frame.shape[:2]
            h, w = frame_shape
            frame_diag = math.hypot(w, h)
            motion = VehicleMotionTracker(frame_diag, config.VEHICLE_MOVE_RATIO_PER_SEC)

        if state.mirror:  # flip before detection so drawn text stays readable
            frame = cv2.flip(frame, 1)

        if raw_slot is not None:
            raw_slot.put(frame.copy())  # untouched view, before drawing

        detections = detector.predict_frame(frame, conf=args.conf, imgsz=args.imgsz)

        # --- hazard layer -------------------------------------------------
        hazard_pairs = find_proximity_hazards(
            detections, frame_diag, config.PROXIMITY_DISTANCE_RATIO)
        moving_vehicles = motion.update(detections, t0)
        workers = [d for d in detections if d.cls == 0]
        elevated = (find_height_hazards(detections, frame.shape[0],
                                        config.HEIGHT_ZONE_FRACTION)
                    if args.height_zone else [])

        active = set()
        if hazard_pairs:
            active.add("proximity")
        if moving_vehicles and workers:
            active.add("vehicle")
        if elevated:
            active.add("height")

        fired = engine.update(
            active,
            detail={"frame": state.frames, "workers": len(workers),
                    "pairs": len(hazard_pairs)},
            messages=compose_alert_messages(hazard_pairs, moving_vehicles,
                                            elevated, workers, frame.shape[1]))

        # --- bookkeeping ----------------------------------------------------
        n_workers = len(workers)
        n_vehicles = sum(1 for d in detections if d.cls == 1)
        dt = time.time() - t0
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        fps_smooth = inst_fps if fps_smooth is None else fps_smooth * 0.9 + inst_fps * 0.1
        with state.lock:
            state.frames += 1
            state.fps = fps_smooth
            state.workers_now = n_workers
            state.vehicles_now = n_vehicles
            state.totals["worker"] += n_workers
            state.totals["dangerous_vehicle"] += n_vehicles
            state.peak["worker"] = max(state.peak["worker"], n_workers)
            state.peak["dangerous_vehicle"] = max(state.peak["dangerous_vehicle"], n_vehicles)
            state.active_hazards = sorted(active)
            if hazard_pairs:
                state.proximity_events.append({
                    "frame": state.frames,
                    "timestamp_sec": round(time.time() - state.start_time, 2),
                    "pairs": len(hazard_pairs),
                })
                del state.proximity_events[:-5000]

        annotated = annotate(frame, detections, sorted(active), hazard_pairs)
        annotated = draw_hud(annotated, state)
        out_slot.put(annotated)

        # on-the-spot reasoning: hand each fired alert's snapshot to the
        # VLM/LLM worker; copy because this loop keeps drawing on frames
        if analyst and fired:
            for event in fired:
                analyst.submit(event, annotated.copy())

    out_slot.close()
    if raw_slot is not None:
        raw_slot.close()


# --------------------------------------------------------------------------
# HTTP server (headless mode)
# --------------------------------------------------------------------------

PWA_MANIFEST = json.dumps({
    "name": "Site Safety Monitor",
    "short_name": "SiteGuard",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#FFFFFF",
    "theme_color": "#155E75",
    "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml"}],
})

PWA_ICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
            '<rect width="100" height="100" rx="18" fill="#155E75"/>'
            '<path d="M50 24 79 74 21 74Z" fill="none" stroke="#fff" '
            'stroke-width="6" stroke-linejoin="round"/>'
            '<line x1="50" y1="44" x2="50" y2="58" stroke="#fff" '
            'stroke-width="6" stroke-linecap="round"/>'
            '<circle cx="50" cy="66" r="3.5" fill="#fff"/></svg>')

# minimal fetch handler makes the app installable when served over HTTPS
PWA_SW = "self.addEventListener('fetch',()=>{});"


def make_handler(out_slot: LatestFrame, state: SessionState,
                 sources: "SourceManager", raw_slot: LatestFrame | None = None):
    dashboard = DASHBOARD_HTML

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence per-request stderr spam
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = dashboard.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/raw"):
                self._stream(raw_slot or out_slot)
            elif self.path.startswith("/stream"):
                self._stream(out_slot)
            elif self.path.startswith("/events"):
                snap = state.snapshot()
                snap["source"] = public_source(sources.current)
                snap["camera_on"] = sources.camera_on
                # never expose internal host/model inventory to viewers
                spark = snap.get("spark") or {}
                snap["spark"] = {"up": spark.get("up", False),
                                 "report_model": spark.get("report_model"),
                                 "vlm_model": spark.get("vlm_model")}
                body = json.dumps(snap).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/switch"):
                self._switch()
            elif self.path == "/manifest.json":
                self._static(PWA_MANIFEST, "application/manifest+json")
            elif self.path == "/icon.svg":
                self._static(PWA_ICON, "image/svg+xml")
            elif self.path == "/sw.js":
                self._static(PWA_SW, "text/javascript")
            elif self.path.startswith("/cameras"):
                from urllib.parse import parse_qs, urlparse
                if parse_qs(urlparse(self.path).query).get("refresh"):
                    _CAM_CACHE["t"] = 0.0  # force re-probe (new device plugged in)
                current = sources.current if isinstance(sources.current, int) else None
                body = json.dumps(list_local_cameras(current)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/camera"):
                self._camera()
            elif self.path.startswith("/report"):
                self._report()
            else:
                self.send_error(404)

        def do_POST(self):
            if not self.path.startswith("/upload"):
                self.send_error(404)
                return
            import re
            import tempfile
            from urllib.parse import parse_qs, urlparse
            name = (parse_qs(urlparse(self.path).query).get("name")
                    or ["upload.mp4"])[0]
            safe = re.sub(r"[^\w.\-]", "_", name)[-80:] or "upload.mp4"
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > 2_000_000_000:
                self.send_error(400, "bad upload size")
                return
            path = Path(tempfile.mkdtemp(prefix="siteguard_")) / safe
            with open(path, "wb") as f:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(65536, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            sources.start(str(path), loop_file=True)  # loops for the demo
            print(f"[live] uploaded video now playing: {safe}")
            body = json.dumps({"ok": True, "source": safe}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _static(self, text: str, content_type: str):
            body = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _camera(self):
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            if "on" in qs:
                on = qs["on"][0] == "1"
                sources.set_camera(on)
                if not on:
                    placeholder = camera_off_frame()
                    if raw_slot is not None:
                        raw_slot.put(placeholder.copy())
                    out_slot.put(placeholder)
                print(f"[live] camera {'ON' if on else 'OFF'}")
            if "auto" in qs:
                with state.lock:
                    state.auto_off = qs["auto"][0] == "1"
            if "mirror" in qs:
                with state.lock:
                    state.mirror = qs["mirror"][0] == "1"
            with state.lock:
                auto, mirror = state.auto_off, state.mirror
            body = json.dumps({"camera_on": sources.camera_on,
                               "auto_off": auto, "mirror": mirror}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _switch(self):
            from urllib.parse import parse_qs, urlparse
            raw = (parse_qs(urlparse(self.path).query).get("src") or [""])[0].strip()
            if raw:
                sources.start(parse_source(raw))
                print(f"[live] source switched to: {raw}")
            body = json.dumps({"ok": bool(raw),
                               "source": public_source(sources.current)}).encode()
            self.send_response(200 if raw else 400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self, slot: LatestFrame):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            seq = 0
            params = [cv2.IMWRITE_JPEG_QUALITY, config.MJPEG_QUALITY]
            last_jpg = None
            with state.lock:
                state.viewers += 1
            try:
                while True:
                    frame, seq = slot.get(seq)
                    if frame is None:
                        if slot.closed:
                            break
                        # idle (camera off / stalled source): resend the last
                        # frame as a keepalive so closed tabs are detected and
                        # the viewer count stays honest
                        if last_jpg is not None:
                            self._send_part(last_jpg)
                        continue
                    # JPEG encoding happens here, in the serving thread,
                    # never in the detection loop
                    ok, jpg = cv2.imencode(".jpg", frame, params)
                    if not ok:
                        continue
                    last_jpg = jpg.tobytes()
                    self._send_part(last_jpg)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # client closed the tab
            finally:
                with state.lock:
                    state.viewers -= 1

        def _send_part(self, jpg_bytes: bytes):
            self.wfile.write(b"--frame\r\n"
                             b"Content-Type: image/jpeg\r\n"
                             b"Content-Length: " + str(len(jpg_bytes)).encode()
                             + b"\r\n\r\n")
            self.wfile.write(jpg_bytes)
            self.wfile.write(b"\r\n")

        def _report(self):
            try:
                from report_generation import generate_hazard_report
                text = generate_hazard_report(state.to_report_dict())
                status = 200
            except Exception as exc:
                text, status = f"Report generation failed: {exc}", 500
            body = text.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def camera_watchdog(state: SessionState, sources: SourceManager,
                    out_slot: LatestFrame, raw_slot: LatestFrame | None,
                    grace_sec: float = 15.0):
    """Meeting-style auto-off: when the last dashboard/stream viewer closes
    and auto-off is enabled, release the camera after a grace period."""
    zero_since = None
    while True:
        time.sleep(5)
        with state.lock:
            viewers, auto = state.viewers, state.auto_off
        if viewers == 0 and auto and sources.camera_on:
            zero_since = zero_since or time.time()
            if time.time() - zero_since >= grace_sec:
                sources.set_camera(False)
                placeholder = camera_off_frame()
                if raw_slot is not None:
                    raw_slot.put(placeholder.copy())
                out_slot.put(placeholder)
                print("[live] no viewers — camera auto-off")
                zero_since = None
        else:
            zero_since = None


def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# --------------------------------------------------------------------------
# Dashboard (single file, no external assets — CSP/offline safe)
# --------------------------------------------------------------------------

DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Site Safety Monitor — Live</title>
<link rel="manifest" href="/manifest.json">
<link rel="icon" href="/icon.svg">
<meta name="theme-color" content="#155E75">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
:root{--cyan:#0891B2;--cyandark:#0E7490;--deep:#155E75;--tint:#EFF7F9;--ink:#122A33;
--line:#C9DDE3;--soft:#DFEBEF;--mut:#4C6B76;--red:#B42318;--redtint:#FCF1EF;
--amber:#B45309;--green:#15803D;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:#FFFFFF;color:var(--ink);
font:14px/1.5 "Barlow","Segoe UI",system-ui,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:12px 18px;background:#fff;
border-top:6px solid var(--deep);border-bottom:1px solid var(--line)}
header .tag{background:var(--deep);color:#fff;font-weight:600;letter-spacing:.22em;
padding:3px 12px;font-size:11px;text-transform:uppercase;border-radius:2px;
font-family:"Barlow Condensed","Segoe UI",sans-serif}
header h1{font-size:17px;letter-spacing:.06em;text-transform:uppercase;font-weight:700;
color:var(--ink);font-family:"Barlow Condensed","Segoe UI",sans-serif}
header .right{margin-left:auto;display:flex;gap:10px;align-items:center;
color:var(--mut);font-size:12px;font-variant-numeric:tabular-nums}
.dot{width:9px;height:9px;border-radius:50%;background:var(--red);display:inline-block}
.dot.ok{background:var(--green)}
main{display:grid;grid-template-columns:minmax(0,2.2fr) minmax(280px,1fr);
gap:14px;padding:14px 18px;max-width:1500px;margin:0 auto}
@media(max-width:900px){main{grid-template-columns:1fr}}
.panel{background:#fff;border:1px solid var(--line);border-radius:2px;overflow:hidden}
.panel h2{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--deep);
padding:10px 12px;border-bottom:1px solid var(--line);font-weight:600;
font-family:"Barlow Condensed","Segoe UI",sans-serif}
.panel h2::before{content:"";display:inline-block;width:9px;height:9px;
background:var(--cyan);margin-right:8px}
#view{width:100%;display:block;background:#000;min-height:280px}
#banner{position:absolute;left:0;right:0;top:0;background:rgba(180,35,24,.94);
color:#fff;text-align:center;font-weight:700;letter-spacing:.14em;padding:10px;
font-size:16px;display:none;text-transform:uppercase;z-index:2}
#banner.on{display:block;animation:blink 1s steps(2) infinite}
@keyframes blink{50%{background:rgba(120,10,5,.94)}}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);
border-top:1px solid var(--line)}
.stat{background:#fff;padding:10px 12px;border-top:3px solid var(--cyan)}
.stat:nth-child(3){border-top-color:var(--red);background:var(--redtint)}
.stat:nth-child(3) .v{color:var(--red)}
.stat .v{font-size:22px;font-weight:700;color:var(--deep);
font-family:"Barlow Condensed","Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
.stat .l{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut)}
.hz{display:flex;flex-direction:column;gap:8px;padding:12px}
.hzrow{display:flex;align-items:center;gap:10px;padding:8px 10px;
border:1px solid var(--line);border-radius:2px;background:#fff}
.hzrow.active{border-color:var(--red);background:var(--redtint)}
.hzrow .n{margin-left:auto;font-weight:700;font-variant-numeric:tabular-nums;
color:var(--deep)}
.hzrow.active .n{color:var(--red)}
.hzrow .k{font-size:12px;text-transform:uppercase;letter-spacing:.08em;
color:var(--deep);font-weight:600}
.hzrow .d{font-size:11px;color:var(--mut)}
#log{list-style:none;max-height:260px;overflow-y:auto;padding:6px 12px;font-size:12px}
#log li{padding:5px 0;border-bottom:1px dashed var(--soft);color:var(--mut)}
#log li b{color:var(--red)}
.foot{display:flex;gap:10px;align-items:center;padding:10px 12px;flex-wrap:wrap}
button{background:transparent;border:1px solid var(--cyan);color:var(--deep);
padding:8px 14px;border-radius:2px;cursor:pointer;font-size:12px;
letter-spacing:.12em;text-transform:uppercase;font-weight:600;
font-family:"Barlow Condensed","Segoe UI",sans-serif}
button:hover{background:var(--tint);border-color:var(--cyandark)}
button.mute.off{border-color:var(--mut);color:var(--mut)}
#reportbtn{background:var(--cyan);color:#fff;border-color:var(--cyandark)}
#reportbtn:hover{background:var(--cyandark)}
#spark{font-size:11px;color:var(--mut)}
#report,#livereport{white-space:pre-wrap;font:12px/1.5 Consolas,monospace;padding:12px;
display:none;max-height:340px;overflow-y:auto;border-top:1px solid var(--line);
background:#fff;color:var(--ink)}
#livereport::before{content:"LIVE SAFETY REPORT (auto-updated on alerts)";display:block;
color:var(--deep);font-weight:700;letter-spacing:.12em;margin-bottom:8px;font-size:11px}
#incidents{list-style:none;max-height:200px;overflow-y:auto;padding:6px 12px;font-size:12px}
#incidents li{padding:6px 0;border-bottom:1px dashed var(--soft)}
#incidents li b{color:var(--amber)}
#incidents li span{color:var(--mut)}
#viewwrap{position:relative;margin:12px;border-radius:10px;overflow:hidden;
background:#0d1418;border:1px solid var(--line)}
#viewtag{position:absolute;right:12px;top:12px;background:rgba(0,0,0,.65);
color:#fff;font-size:10px;letter-spacing:.16em;padding:3px 9px;border-radius:999px}
#namechip{position:absolute;left:14px;top:12px;color:#fff;font-size:13px;
text-shadow:0 1px 3px rgba(0,0,0,.9);max-width:55%;overflow:hidden;
text-overflow:ellipsis;white-space:nowrap}
#vidctl{position:absolute;left:50%;bottom:14px;transform:translateX(-50%);
display:flex;gap:14px;z-index:3}
.circ{width:48px;height:48px;border-radius:50%;border:1px solid rgba(255,255,255,.35);
background:rgba(25,34,40,.85);color:#fff;font-size:19px;cursor:pointer;
display:flex;align-items:center;justify-content:center;padding:0;
letter-spacing:0;text-transform:none}
.circ:hover{background:rgba(45,60,68,.95)}
.circ.off{background:#d93025;border-color:#d93025}
.circ.act{background:var(--cyan);border-color:var(--cyan)}
#pills{display:flex;gap:10px;padding:2px 14px 12px;flex-wrap:wrap;align-items:center;
justify-content:center}
.pill,.pillwrap{background:#fff;border:1px solid var(--line);color:var(--ink);
border-radius:999px;font-size:12.5px;
letter-spacing:0;text-transform:none;font-family:"Barlow","Segoe UI",sans-serif;
font-weight:500}
.pill{padding:9px 16px;cursor:pointer}
.pillwrap{display:flex;align-items:center;gap:8px;padding:0 6px 0 14px;min-height:38px}
.pillwrap svg,.iconbtn svg{width:16px;height:16px;fill:none;stroke:var(--mut);
stroke-width:2;stroke-linecap:round;stroke-linejoin:round;flex:0 0 auto}
.pillwrap select,.pillwrap input{border:none;background:none;outline:none;
color:var(--ink);font:inherit;padding:8px 6px 8px 0;min-width:0}
.pillwrap select{cursor:pointer;max-width:230px}
.pillwrap.grow{flex:1;min-width:190px}
.pillwrap.grow input{flex:1;width:100%}
.pillwrap:focus-within,.pillwrap:hover,button.pill:hover{border-color:var(--cyan);
background:var(--tint)}
.iconbtn{width:38px;height:38px;border-radius:50%;border:1px solid var(--line);
background:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;
padding:0}
.iconbtn:hover{border-color:var(--cyan);background:var(--tint)}
.iconbtn.spin svg{animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
button.pill svg{width:15px;height:15px;fill:none;stroke:currentColor;stroke-width:2;
stroke-linecap:round;stroke-linejoin:round;vertical-align:-2px;margin-right:7px}
.circ svg{width:22px;height:22px;fill:none;stroke:#fff;stroke-width:2;
stroke-linecap:round;stroke-linejoin:round}
#srcnow{font-size:11px;color:var(--mut);width:100%;text-align:center}
#autolbl{font-size:11px;color:var(--mut);display:flex;align-items:center;gap:5px;
cursor:pointer}
#autolbl input{accent-color:var(--cyan)}
@media(max-width:640px){
  header{padding:10px 12px;gap:8px}
  header h1{font-size:14px;letter-spacing:.04em}
  main{padding:10px;gap:10px}
  #viewwrap{margin:8px;border-radius:8px}
  #view{min-height:200px}
  .stats{grid-template-columns:repeat(2,1fr)}
  #pills{padding:2px 10px 10px;gap:8px}
  .pillwrap{min-height:44px}
  .pillwrap select{max-width:none;flex:1}
  .pillwrap,.pillwrap:not(.grow){flex:1 1 100%}
  .iconbtn{width:44px;height:44px;flex:0 0 auto}
  .circ{width:52px;height:52px}
  button{min-height:40px}
}
</style></head><body>
<header>
  <span class="tag">Site Safety Monitor</span>
  <h1>Live Hazard Detection</h1>
  <div class="right">
    <span id="fps" class="stat-inline">— fps</span>
    <span class="dot" id="livedot"></span>
  </div>
</header>
<main>
  <section class="panel">
    <div id="viewwrap">
      <div id="banner">HAZARD</div>
      <img id="view" src="/stream.mjpg" alt="live stream">
      <span id="namechip">Site camera</span>
      <span id="viewtag">ANNOTATED</span>
      <div id="vidctl">
        <button class="circ" id="cambtn" title="Turn camera on/off"></button>
        <button class="circ" id="flipbtn" title="Mirror / flip camera"></button>
        <button class="circ" id="mutebtn" title="Voice alerts on/off"></button>
      </div>
    </div>
    <div id="pills">
      <label class="pillwrap" title="Camera device">
        <svg viewBox="0 0 24 24"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
        <select id="camsel"><option value="">scanning cameras…</option></select>
      </label>
      <button class="iconbtn" id="rescan" title="Rescan for new cameras">
        <svg viewBox="0 0 24 24"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
      </button>
      <label class="pillwrap" title="View">
        <svg viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
        <select id="viewsel">
          <option value="/stream.mjpg" data-tag="ANNOTATED">Annotated view</option>
          <option value="/raw.mjpg" data-tag="DIRECT">Direct view</option>
        </select>
      </label>
      <button class="pill" id="upbtn" title="Run detection on a video file (loops)">
        <svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        Upload video</button>
      <input type="file" id="upfile" accept="video/*" hidden>
      <span id="srcnow"></span>
    </div>
    <div class="stats">
      <div class="stat"><div class="v" id="workers">0</div><div class="l">Workers</div></div>
      <div class="stat"><div class="v" id="vehicles">0</div><div class="l">Vehicles</div></div>
      <div class="stat"><div class="v" id="alerts">0</div><div class="l">Alerts fired</div></div>
      <div class="stat"><div class="v" id="uptime">0s</div><div class="l">Uptime</div></div>
    </div>
  </section>
  <section class="panel">
    <h2>Hazard use cases</h2>
    <div class="hz">
      <div class="hzrow" id="hz-proximity">
        <div><div class="k">Proximity</div><div class="d">Worker too close to vehicle</div></div>
        <span class="n" id="n-proximity">0</span></div>
      <div class="hzrow" id="hz-vehicle">
        <div><div class="k">Vehicle</div><div class="d">Heavy vehicle moving in work zone</div></div>
        <span class="n" id="n-vehicle">0</span></div>
      <div class="hzrow" id="hz-height">
        <div><div class="k">Height</div><div class="d">Worker at height near edge (experimental)</div></div>
        <span class="n" id="n-height">0</span></div>
    </div>
    <h2>Alert log</h2>
    <ul id="log"><li>No alerts yet.</li></ul>
    <h2>On-the-spot AI analysis</h2>
    <ul id="incidents"><li>Scene notes appear here seconds after an alert fires.</li></ul>
    <h2>Safety report</h2>
    <div class="foot">
      <button id="reportbtn">Generate report</button>
      <label id="autolbl"><input type="checkbox" id="autooff" checked>
        auto-off camera when everyone closes</label>
      <label id="autolbl"><input type="checkbox" id="detailvoice" checked>
        speak AI scene details after alarms</label>
      <span id="spark">Spark: checking…</span>
    </div>
    <div id="livereport"></div>
    <div id="report"></div>
  </section>
</main>
<script>
let muted=false,lastAlert=0,alertTotal=0,camOn=true,camList=[],lastIncKey=null;
const $=id=>document.getElementById(id);
const IC={
cam:'<svg viewBox="0 0 24 24"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>',
camOff:'<svg viewBox="0 0 24 24"><path d="M16 16v1a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h2m5.66 0H14a2 2 0 0 1 2 2v3.34l1 1L23 7v10"/><line x1="1" y1="1" x2="23" y2="23"/></svg>',
vol:'<svg viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>',
volOff:'<svg viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>',
flip:'<svg viewBox="0 0 24 24"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>'};
$('cambtn').innerHTML=IC.cam;
$('mutebtn').innerHTML=IC.vol;
$('flipbtn').innerHTML=IC.flip;
let mirrored=false;
$('flipbtn').onclick=async()=>{
  try{const r=await fetch('/camera?mirror='+(mirrored?0:1));const d=await r.json();
    mirrored=d.mirror;
    $('flipbtn').classList.toggle('act',mirrored);}
  catch(e){}};
$('mutebtn').onclick=()=>{muted=!muted;
  $('mutebtn').innerHTML=muted?IC.volOff:IC.vol;
  $('mutebtn').classList.toggle('off',muted);
  if(!muted)say('Voice alerts enabled');};
function say(text,queue){
  if(muted||!window.speechSynthesis)return;
  const u=new SpeechSynthesisUtterance(text);
  u.rate=1.05;u.pitch=1;u.volume=1;
  if(!queue)speechSynthesis.cancel(); // alarms preempt; details wait their turn
  speechSynthesis.speak(u);}
function fmtUp(s){return s>=3600?(s/3600).toFixed(1)+'h':s>=60?(s/60).toFixed(0)+'m':s.toFixed(0)+'s';}
async function poll(){
  try{
    const r=await fetch('/events');const d=await r.json();
    $('livedot').classList.add('ok');
    $('fps').textContent=d.fps+' fps';
    $('workers').textContent=d.workers;
    $('vehicles').textContent=d.vehicles;
    $('uptime').textContent=fmtUp(d.uptime_sec);
    let total=0;
    for(const k of['proximity','vehicle','height']){
      const c=d.alert_counts[k]||0;total+=c;
      $('n-'+k).textContent=c;
      $('hz-'+k).classList.toggle('active',d.active_hazards.includes(k));}
    $('alerts').textContent=total;
    const banner=$('banner');
    if(d.active_hazards.length){
      banner.textContent='⚠ '+d.active_hazards.map(k=>d.hazard_labels[k]||k).join('  •  ');
      banner.classList.add('on');
    }else banner.classList.remove('on');
    if(d.recent_alerts.length){
      const log=$('log');log.innerHTML='';
      [...d.recent_alerts].reverse().forEach(a=>{
        const li=document.createElement('li');
        li.innerHTML='<b>'+a.type.toUpperCase()+'</b> — '+a.message+' <i>('+a.iso+')</i>';
        log.appendChild(li);});
      const newest=d.recent_alerts[d.recent_alerts.length-1];
      if(newest.time>lastAlert){
        if(lastAlert>0)say(newest.message); // don't replay history on page load
        lastAlert=newest.time;}
      else if(lastAlert===0)lastAlert=newest.time;
    }
    if(d.incidents&&d.incidents.length){
      const inc=$('incidents');inc.innerHTML='';
      [...d.incidents].reverse().forEach(n=>{
        const li=document.createElement('li');
        li.innerHTML='<b>'+(n.type||'').toUpperCase()+'</b> <span>'+n.iso+'</span><br>'
                     +(n.note||'');
        inc.appendChild(li);});
      // follow-up voice: read the VLM scene note aloud once it lands
      // (queued, so it never cuts off the instant alarm phrase)
      const nw=d.incidents[d.incidents.length-1];
      const key=nw.iso+nw.type+(nw.note||'');
      if(lastIncKey!==null&&key!==lastIncKey&&$('detailvoice').checked
         &&nw.note&&nw.note!=='scene review unavailable')
        say('Detail: '+nw.note.slice(0,220),true);
      lastIncKey=key;
    }
    if(d.source!==undefined){
      const digit=/^[0-9]+$/.test(d.source);
      const cam=digit?camList.find(c=>String(c.index)===d.source):null;
      const label=cam?cam.label:(digit?'Camera '+d.source
        :d.source.split(/[\\/]/).pop());
      $('namechip').textContent=(d.camera_on?'':'OFF · ')+label;
      if(digit&&document.activeElement!==$('camsel'))$('camsel').value=d.source;
      if(!$('srcnow').textContent.startsWith('Switching'))
        $('srcnow').textContent=(d.camera_on?'live':'camera off')
          +'  ·  viewers: '+d.viewers;}
    if(d.camera_on!==undefined){
      camOn=d.camera_on;
      $('cambtn').innerHTML=camOn?IC.cam:IC.camOff;
      $('cambtn').classList.toggle('off',!camOn);}
    if(d.auto_off!==undefined&&document.activeElement!==$('autooff'))
      $('autooff').checked=d.auto_off;
    if(d.mirror!==undefined){mirrored=d.mirror;
      $('flipbtn').classList.toggle('act',mirrored);}
    if(d.live_report){
      const lr=$('livereport');
      if(lr.textContent!==d.live_report){lr.textContent=d.live_report;}
      lr.style.display='block';
    }
    const s=d.spark;
    $('spark').textContent=s.up
      ?'AI reports: '+(s.report_model||'connected')
      :'AI reports: offline — using template fallback';
  }catch(e){$('livedot').classList.remove('ok');}
}
async function switchSrc(v){
  if(!v)return;
  $('srcnow').textContent='Switching to '+v+' …';
  try{const r=await fetch('/switch?src='+encodeURIComponent(v));const d=await r.json();
    $('srcnow').textContent='Current source: '+d.source;}
  catch(e){$('srcnow').textContent='Switch failed: '+e;}}
async function loadCams(refresh){
  if(refresh)$('rescan').classList.add('spin');
  try{const r=await fetch('/cameras'+(refresh?'?refresh=1':''));
    const fresh=await r.json();
    // announce newly plugged devices (e.g. RealSense connected mid-session)
    if(camList.length){
      const known=new Set(camList.map(c=>c.index));
      fresh.filter(c=>!known.has(c.index)).forEach(c=>{
        $('srcnow').textContent='New camera detected: '+c.label;
        say('New camera detected: '+c.label,true);});}
    camList=fresh;
    const sel=$('camsel');const prev=sel.value;sel.innerHTML='';
    camList.forEach(c=>{const o=document.createElement('option');
      o.value=String(c.index);o.textContent=c.label;sel.appendChild(o);});
    if(!camList.length){const o=document.createElement('option');
      o.value='';o.textContent='no local camera found';sel.appendChild(o);}
    if(prev&&[...sel.options].some(o=>o.value===prev))sel.value=prev;
  }catch(e){}
  $('rescan').classList.remove('spin');}
loadCams();
setInterval(()=>loadCams(true),20000); // auto-detect newly plugged cameras
$('rescan').onclick=()=>loadCams(true);
$('upbtn').onclick=()=>$('upfile').click();
$('upfile').onchange=()=>{
  const f=$('upfile').files[0];if(!f)return;
  const xhr=new XMLHttpRequest();
  xhr.open('POST','/upload?name='+encodeURIComponent(f.name));
  xhr.upload.onprogress=e=>{if(e.lengthComputable)
    $('srcnow').textContent='Uploading '+f.name+' — '
      +Math.round(e.loaded/e.total*100)+'%';};
  xhr.onload=()=>{$('srcnow').textContent='Detecting on uploaded video (loops): '+f.name;};
  xhr.onerror=()=>{$('srcnow').textContent='Upload failed.';};
  $('srcnow').textContent='Uploading '+f.name+' …';
  xhr.send(f);
  $('upfile').value='';};
$('camsel').onchange=()=>{if($('camsel').value!=='')switchSrc($('camsel').value);};
$('viewsel').onchange=()=>{
  const o=$('viewsel').selectedOptions[0];
  $('view').src=o.value;
  $('viewtag').textContent=o.dataset.tag;};
$('cambtn').onclick=async()=>{
  try{const r=await fetch('/camera?on='+(camOn?0:1));const d=await r.json();
    camOn=d.camera_on;
    $('cambtn').innerHTML=camOn?IC.cam:IC.camOff;
    $('cambtn').classList.toggle('off',!camOn);}
  catch(e){}};
$('autooff').onchange=()=>fetch('/camera?auto='+($('autooff').checked?1:0));
$('reportbtn').onclick=async()=>{
  const box=$('report');box.style.display='block';
  box.textContent='Generating safety report on the Spark…';
  try{const r=await fetch('/report');box.textContent=await r.text();}
  catch(e){box.textContent='Report request failed: '+e;}};
setInterval(poll,1000);poll();
if('serviceWorker' in navigator)navigator.serviceWorker.register('/sw.js').catch(()=>{});
</script>
</body></html>
"""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Real-time hazard detection")
    p.add_argument("--weights", default=str(config.DEFAULT_WEIGHTS),
                   help="Plan A YOLO weights (best.pt)")
    p.add_argument("--source", default="0",
                   help="Camera index or RTSP/HTTP stream URL (default 0)")
    p.add_argument("--video", default=None,
                   help="Video file instead of a camera; loops forever")
    p.add_argument("--mock", action="store_true",
                   help="Synthetic detector — rehearse with no GPU/weights")
    p.add_argument("--headless", action="store_true",
                   help="No window; serve MJPEG dashboard over HTTP")
    p.add_argument("--port", type=int, default=config.DASHBOARD_PORT)
    p.add_argument("--conf", type=float, default=config.LIVE_CONF)
    p.add_argument("--imgsz", type=int, default=config.LIVE_IMGSZ)
    p.add_argument("--device", default="0", help="CUDA device or 'cpu'")
    p.add_argument("--no-audio", action="store_true",
                   help="Disable server-side speaker output")
    p.add_argument("--no-analysis", action="store_true",
                   help="Disable on-the-spot VLM/LLM incident analysis")
    p.add_argument("--height-zone", action="store_true",
                   default=config.HEIGHT_ZONE_ENABLED,
                   help="Enable experimental height-hazard heuristic")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mock and args.device == "0":
        args.device = "cpu"

    detector = build_detector(args)
    print(f"[live] detector: {getattr(detector, 'label', 'Plan A')}")
    print(f"[live] Ollama (Spark): {config.OLLAMA_HOST}")

    player = None if args.no_audio else AudioPlayer()
    engine = AlertEngine(player=player)
    state = SessionState(engine)
    state.detector_label = getattr(detector, "label", "Plan A — Fine-tuned YOLOv8")

    source = args.video if args.video else parse_source(args.source)
    in_slot, out_slot, raw_slot = LatestFrame(), LatestFrame(), LatestFrame()
    sources = SourceManager(in_slot)
    sources.start(source, loop_file=bool(args.video), critical=True)

    threading.Thread(target=spark_status_poller, args=(state,), daemon=True).start()

    analyst = None
    if not args.no_analysis:
        analyst = IncidentAnalyst(state)
        analyst.start()

    stop_flag = threading.Event()
    det_thread = threading.Thread(
        target=detection_loop,
        args=(args, detector, in_slot, out_slot, state, engine, stop_flag, analyst,
              raw_slot),
        daemon=True)
    det_thread.start()

    if not args.headless:
        # opencv-python-headless has no GUI: fall back to the web dashboard
        # instead of dying silently
        try:
            cv2.namedWindow("_gui_probe")
            cv2.destroyWindow("_gui_probe")
        except cv2.error:
            print("[live] this OpenCV build has no display support — "
                  "switching to the web dashboard (--headless)")
            args.headless = True

    server = None
    try:
        if args.headless:
            threading.Thread(target=camera_watchdog,
                             args=(state, sources, out_slot, raw_slot),
                             daemon=True).start()
            server = ThreadingHTTPServer(("0.0.0.0", args.port),
                                         make_handler(out_slot, state, sources,
                                                      raw_slot))
            print(f"[live] dashboard:  http://{local_ip()}:{args.port}")
            print(f"[live] MJPEG feed: http://{local_ip()}:{args.port}/stream.mjpg")
            print("[live] Ctrl+C to stop")
            server.serve_forever()
        else:
            seq = 0
            while det_thread.is_alive():
                frame, seq = out_slot.get(seq)
                if frame is None:
                    if out_slot.closed:
                        break
                    continue
                cv2.imshow("Site Safety Monitor — press q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        sources.stop()
        if server:
            server.shutdown()
        print(f"\n[live] session summary: {json.dumps(state.snapshot()['alert_counts'])}")
        print(f"[live] events log: {config.EVENTS_LOG}")


if __name__ == "__main__":
    main()
