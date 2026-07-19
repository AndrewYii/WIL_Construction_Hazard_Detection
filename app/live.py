"""
Real-time construction hazard detection with on-the-spot audio alerts.

Pipeline (all threads, never blocking each other):
  capture thread  -> latest-frame slot (stale frames dropped)
  detection loop  -> YOLO -> hazard layer (proximity / vehicle / height / ppe)
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
from hazard_logic import (Detection, VehicleMotionTracker, boxes_overlap,
                          find_height_hazards, find_ppe_violations,
                          find_proximity_hazards)

FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_COLORS = {0: (80, 220, 60), 1: (0, 165, 255)}  # BGR: workers green, vehicles orange
CLASS_NAMES = {0: "worker", 1: "vehicle"}
ALERT_BG = (30, 30, 220)
PPE_VIOLATION_COLOR = (0, 0, 255)  # BGR pure red — overrides the normal worker green

HAZARD_LABELS = {
    "proximity": "WORKER TOO CLOSE TO VEHICLE",
    "vehicle": "VEHICLE MOVING IN WORK ZONE",
    "height": "WORKER AT HEIGHT",
    "ppe": "PPE NOT WORN",
}


_CAM_CACHE = {"t": 0.0, "list": []}

# V4L2 ioctl constants (Linux, architecture-independent — videodev2.h).
# QUERYCAP/ENUM_FMT are metadata-only queries: they never start streaming,
# so unlike opening the device for capture they don't light up the camera's
# LED or interfere with an active capture elsewhere.
_VIDIOC_QUERYCAP = 0x80685600
_VIDIOC_ENUM_FMT = 0xC0405602
_V4L2_CAP_VIDEO_CAPTURE = 0x00000001
_V4L2_CAP_DEVICE_CAPS = 0x80000000
_V4L2_BUF_TYPE_VIDEO_CAPTURE = 1


def _v4l2_probe(index: int) -> tuple[bool, str]:
    """(can_capture, format_description) for /dev/videoN, via metadata-only
    ioctls. A single physical camera (RealSense, some depth/IR modules, some
    webcams with a separate metadata interface) can expose SEVERAL /dev/videoN
    nodes sharing the exact same name — most of them not actually usable for
    capture, which otherwise makes the dashboard's camera list show several
    identical, mostly-broken entries. can_capture=False for those; the format
    description (e.g. "YUYV 4:2:2", "16-bit Depth") disambiguates the rest."""
    import fcntl
    import os
    import struct
    try:
        fd = os.open(f"/dev/video{index}", os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return False, ""
    try:
        cap = bytearray(104)
        fcntl.ioctl(fd, _VIDIOC_QUERYCAP, cap)
        caps, device_caps = struct.unpack_from("<II", cap, 84)
        effective = device_caps if (caps & _V4L2_CAP_DEVICE_CAPS) else caps
        if not (effective & _V4L2_CAP_VIDEO_CAPTURE):
            return False, ""
        desc = ""
        fmt = bytearray(64)
        struct.pack_into("<II", fmt, 0, 0, _V4L2_BUF_TYPE_VIDEO_CAPTURE)
        try:
            fcntl.ioctl(fd, _VIDIOC_ENUM_FMT, fmt)
            desc = fmt[12:44].split(b"\0", 1)[0].decode(errors="replace")
        except OSError:
            pass
        return True, desc
    except OSError:
        return False, ""
    finally:
        os.close(fd)


def list_local_cameras(current=None, max_probe: int = 5) -> list[dict]:
    """Meet-style device list WITHOUT opening any camera (no LED flicker,
    no interference with an active capture).

    Windows: DirectShow's own device list (pygrabber) — the order exactly
    matches cv2.CAP_DSHOW indices, so labels can't be swapped.
    Linux: /sys/class/video4linux/videoN/name — N is the OpenCV index, then
    filtered/disambiguated via _v4l2_probe for cameras exposing multiple
    identically-named nodes.
    Fallback: probe indices by opening them (old behavior). Cached 15s."""
    import platform
    now = time.time()
    if now - _CAM_CACHE["t"] < 15 and _CAM_CACHE["list"]:
        return _CAM_CACHE["list"]
    system = platform.system()
    cams: list[dict] = []
    try:
        if system == "Windows":
            import comtypes
            from pygrabber.dshow_graph import FilterGraph
            comtypes.CoInitialize()  # handler threads need their own COM init
            try:
                names = FilterGraph().get_input_devices()
            finally:
                comtypes.CoUninitialize()
            cams = [{"index": i, "label": n, "usable": True} for i, n in enumerate(names)]
        else:
            import glob
            import re as _re
            raw = []
            for path in sorted(glob.glob("/sys/class/video4linux/video*/name")):
                # anchor to the trailing "videoN/name" — a naive "video(\d+)"
                # search matches "video4" out of "video4linux" first and
                # reports every camera as index 4 (a real bug found 2026-07-19)
                m = _re.search(r"video(\d+)/name$", path)
                if not m:
                    continue
                with open(path) as f:
                    raw.append({"index": int(m.group(1)), "label": f.read().strip()})
            dupes = {c["label"] for c in raw
                     if sum(1 for o in raw if o["label"] == c["label"]) > 1}
            for c in raw:
                can_capture, desc = _v4l2_probe(c["index"])
                if not can_capture:
                    continue  # metadata-only node — not something cv2 can stream
                c["usable"] = True
                if c["label"] in dupes:
                    # Some multi-sensor cameras (RealSense: depth+IR+color)
                    # expose several capture-capable nodes under one name —
                    # flag the ones that are unambiguously not color. Labeling
                    # alone wasn't foolproof enough (2026-07-19: a user still
                    # picked the depth node after it was clearly marked "not
                    # usable" — three identical-looking entries invite a
                    # misclick regardless of label text, and RealSense's own
                    # USB flakiness can renumber which index is which between
                    # reconnects). `usable` lets the dashboard make the wrong
                    # ones genuinely unclickable instead of just labeled.
                    mono = any(k in desc.lower() for k in
                              ("depth", "grey", "gray", "infrared"))
                    suffix = " — not usable here, no color" if mono else " (recommended)"
                    c["label"] = f"{c['label']} — {desc or 'unknown format'}{suffix} (video{c['index']})"
                    c["usable"] = not mono
                cams.append(c)
            cams.sort(key=lambda c: not c["usable"])
    except Exception:
        cams = []
    if not cams:  # fallback: probe by opening (may blink camera LEDs)
        backend = cv2.CAP_DSHOW if system == "Windows" else cv2.CAP_ANY
        for i in range(max_probe):
            if current is not None and i == current:
                cams.append({"index": i, "label": f"Camera {i}", "usable": True})
                continue
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                cams.append({"index": i, "label": f"Camera {i}", "usable": True})
            cap.release()
    _CAM_CACHE.update(t=now, list=cams)
    return cams


def camera_off_frame(text: str = "CAMERA OFF", shape=(360, 640, 3)):
    """Placeholder pushed to the streams when the camera is off, so viewers
    see an explicit OFF state instead of a frozen last frame."""
    import numpy as np
    frame = np.zeros(shape, dtype=np.uint8)
    h, w = shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.9, 2)
    cv2.putText(frame, text, (max((w - tw) // 2, 10), (h + th) // 2), FONT, 0.9,
                (120, 120, 120), 2)
    return frame


def load_tts_voice():
    """Server-side neural voice (Piper) for alerts — every viewer hears the
    same natural voice regardless of what's installed on their own device.
    Loaded once at startup (~0.7s); synthesis itself is ~150-200ms for a
    short alert sentence, well inside "on the spot". Returns None (and the
    dashboard silently falls back to each browser's own Web Speech API) if
    disabled or the voice model hasn't been downloaded — never blocks
    startup or crashes the dashboard over a missing/optional voice."""
    if not config.TTS_ENABLED:
        return None
    if not config.TTS_VOICE_PATH.exists():
        print(f"[live] TTS voice not found at {config.TTS_VOICE_PATH} — "
              "falling back to each browser's own voice. Get one with: "
              "python -m piper.download_voices --download-dir "
              "assets/tts_voices en_US-ryan-medium")
        return None
    try:
        from piper import PiperVoice
        voice = PiperVoice.load(str(config.TTS_VOICE_PATH))
        print(f"[live] TTS voice loaded: {config.TTS_VOICE_PATH.name}")
        return voice
    except Exception as exc:
        print(f"[live] TTS voice failed to load ({exc}) — falling back to "
              "each browser's own voice")
        return None


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
        import platform
        # match the backend used when enumerating cameras, or Windows index
        # numbers won't line up (MSMF and DSHOW order devices differently)
        backend = (cv2.CAP_DSHOW
                   if platform.system() == "Windows" and isinstance(self.source, int)
                   else cv2.CAP_ANY)
        cap = None
        for _ in range(6):  # a just-released device can take a moment to free
            if self.stop_flag.is_set():
                return
            cap = cv2.VideoCapture(self.source, backend)
            if cap.isOpened():
                break
            cap.release()
            cap = None
            time.sleep(0.5)
        if cap is None:
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
            if self.stop_flag.is_set():
                break  # don't overwrite the CAMERA OFF placeholder on shutdown
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

    def arm(self, source, loop_file: bool = False):
        """Remember the source without opening the device (standby): the
        camera stays off until someone presses the camera button."""
        with self._lock:
            self.current = source
            self._loop_file = loop_file
            self.camera_on = False

    def use_browser(self):
        """A viewer's device camera becomes the source: frames arrive over
        HTTP (/ingest) instead of a local capture thread."""
        with self._lock:
            if self.capture and self.capture.is_alive():
                self.capture.close_on_exit = False
                self.capture.stop_flag.set()
            self.capture = None
            self.current = "browser"
            self.camera_on = True

    def set_camera(self, on: bool):
        """Meeting-style camera toggle: off releases the device entirely
        (privacy — its LED goes dark), on reopens the remembered source."""
        if on:
            if self.current == "browser":
                self.camera_on = True  # frames resume from the sender
            elif not self.camera_on and self.current is not None:
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

    @property
    def is_file_source(self) -> bool:
        """True while the current source is a looping uploaded video, not a
        live camera/RTSP/browser feed. The meeting-style privacy auto-off
        exists to dim a webcam's LED when nobody's watching — it must not
        also kill an uploaded video's playback just because the tab was
        briefly closed (e.g. mid page-refresh), which used to freeze the
        stream on a permanent "camera off" placeholder with no way back
        short of manually toggling the camera button."""
        return self._loop_file


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
        self.height_flash_until = 0.0     # banner window after verified height alert
        self.session_epoch = 0            # bumped by reset_session() — see IncidentAnalyst

    def reset_session(self):
        """A new source (upload, camera switch, or a tab change that stops
        the previous one) starts a fresh session — old counts/log/report
        mixed with a totally different video or camera is misleading, not
        historical record-keeping (2026-07-19). Only resets what the
        dashboard displays live; logs/events.jsonl (the permanent audit
        trail) is untouched — this is a display reset, not data deletion."""
        with self.lock:
            self.start_time = time.time()
            self.frames = 0
            self.fps = 0.0
            self.workers_now = 0
            self.vehicles_now = 0
            self.peak = {"worker": 0, "dangerous_vehicle": 0}
            self.totals = {"worker": 0, "dangerous_vehicle": 0}
            self.active_hazards = []
            self.proximity_events = []
            self.incidents = []
            self.live_report = ""
            self.height_flash_until = 0.0
            self.session_epoch += 1  # any in-flight analyst job from before this must be discarded
        self.engine.reset()

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


def parse_ppe(text: str | None) -> dict | None:
    """Extract {'hardhat': bool, 'harness': bool} from a VLM reply, or None
    if the reply is unusable (treated as unverified → alert fires anyway)."""
    if not text:
        return None
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return {"hardhat": bool(parsed.get("hardhat")),
                "harness": bool(parsed.get("harness"))}
    except Exception:
        return None


def crop_person(frame, det, margin: float = 0.35):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = det.xyxy
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    a, b = max(int(x1 - mx), 0), max(int(y1 - my), 0)
    c, d = min(int(x2 + mx), w), min(int(y2 + my), h)
    return frame[b:d, a:c].copy()


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

    PPE_PROMPT = (
        "This cropped image shows a construction worker who appears to be "
        "working at height. Check fall-protection equipment. Respond ONLY "
        'with compact JSON: {"hardhat": true/false, "harness": true/false}. '
        "harness means a visible safety harness, lanyard, or fall-arrest strap."
    )

    def __init__(self, state: SessionState, engine: AlertEngine | None = None,
                 report_refresh_sec: float = config.REPORT_REFRESH_SEC,
                 ppe_detector=None):
        super().__init__(daemon=True)
        self.state = state
        self.engine = engine
        self.report_refresh_sec = report_refresh_sec
        self.ppe_detector = ppe_detector
        self._jobs: queue.Queue = queue.Queue(maxsize=4)
        self._last_report = 0.0

    def submit(self, event: dict, frame):
        try:
            self._jobs.put_nowait(("incident", event, frame, self.state.session_epoch))
        except queue.Full:
            pass  # analysis is best-effort; the alert itself already fired

    def submit_height(self, message: str, crop):
        """Verify a geometric height candidate: alert only if the worker has
        no visible fall protection (or if verification is impossible)."""
        try:
            self._jobs.put_nowait(("height", message, crop, self.state.session_epoch))
        except queue.Full:
            pass

    def _check_ppe_detector(self, crop) -> dict | None:
        """Run the local PPE_Detect model on the height crop. Returns
        {'hardhat': bool} or None if the detector is unavailable/disabled —
        None counts as "not confirmed", same as an unparseable VLM verdict."""
        if self.ppe_detector is None:
            return None
        try:
            dets = self.ppe_detector.predict(crop, conf=config.PPE_CONF, imgsz=320)
        except Exception:
            return None
        has_hardhat = any(d["name"] == "Hardhat" for d in dets)
        has_no_hardhat = any(d["name"] == "NO-Hardhat" for d in dets)
        return {"hardhat": has_hardhat and not has_no_hardhat}

    def _verify_height(self, client, message: str, crop, epoch: int):
        # This VLM round trip can take several seconds — if a tab switch or
        # new upload started a new session while it was running (found
        # 2026-07-19: a stale height verdict fired an alert well after the
        # user had already switched away from the video it was about), the
        # result belongs to a session that no longer exists. Bail out before
        # doing the (wasted) work if it's already stale.
        if epoch != self.state.session_epoch:
            return
        verdict = None
        try:
            ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok and client.is_up():
                verdict = parse_ppe(client.describe_image(self.PPE_PROMPT,
                                                          jpg.tobytes()))
        except Exception:
            verdict = None
        if epoch != self.state.session_epoch:
            return  # session moved on while the VLM call was in flight
        ppe_check = self._check_ppe_detector(crop)
        # Two independent signals required to suppress a real fall-risk
        # alarm: the VLM must see the harness itself (the item that actually
        # matters — no local detector here has that class), AND PPE_Detect
        # must independently confirm a hardhat on the same crop. Either one
        # missing/unconfirmed/disagreeing → alarm fires (safety-first). If
        # PPE_Detect is unavailable (weights missing / PPE_ENABLED=0), the
        # cross-check can never pass, so height alarms always fire.
        protected = bool(verdict and verdict["harness"]
                         and ppe_check and ppe_check["hardhat"])
        if protected:
            note = ("Height check: worker elevated, harness visible (VLM) and "
                    "hardhat confirmed (PPE_Detect) — alarm suppressed.")
        else:
            if verdict is None:
                message += " PPE could not be verified."
            elif not verdict["harness"]:
                message += " No harness visible."
            elif not (ppe_check and ppe_check["hardhat"]):
                message += " Hardhat not confirmed."
            fired = self.engine.fire_now("height", message) if self.engine else None
            note = ("Height ALERT: fall protection not fully confirmed on "
                    f"elevated worker (harness={bool(verdict and verdict['harness'])}, "
                    f"hardhat={bool(ppe_check and ppe_check['hardhat'])})." if fired else
                    "Height: elevated without confirmed PPE, alarm cooling down.")
            if fired:
                with self.state.lock:
                    self.state.height_flash_until = time.time() + 6
        with self.state.lock:
            self.state.incidents.append({
                "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "height", "note": note})
            del self.state.incidents[:-50]

    def run(self):
        from llm_client import get_client
        from report_generation import generate_hazard_report
        client = get_client()
        while True:
            kind, event, frame, epoch = self._jobs.get()
            if kind == "height":
                self._verify_height(client, event, frame, epoch)
                continue
            if epoch != self.state.session_epoch:
                continue  # stale — a new session started while this was queued
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
            if epoch != self.state.session_epoch:
                continue  # session moved on while the VLM call was in flight
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


def annotate(frame, detections, hazard_types, hazard_pairs=None, ppe_violations=None):
    ppe_violations = ppe_violations or []
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.xyxy]
        # A PPE-violating worker gets its own red box + label, on top of
        # (not instead of) the normal box color — a red worker box says
        # exactly who's non-compliant at a glance, not just that PPE is
        # some hazard somewhere in frame. ppe_violations is held over from
        # the last throttled PPE_Detect pass (up to PPE_CHECK_INTERVAL_SEC
        # old) — its Detection objects are stale, never equal to this
        # frame's freshly re-detected ones (conf/xyxy jitter every frame),
        # so match by spatial overlap, not identity/equality.
        violator = any(boxes_overlap(det.xyxy, v.xyxy) for v in ppe_violations)
        color = PPE_VIOLATION_COLOR if violator else BOX_COLORS.get(det.cls, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if violator else 2)
        label = f"{CLASS_NAMES.get(det.cls, det.cls)} {det.conf:.2f}"
        if violator:
            label += " — NO PPE"
        cv2.putText(frame, label, (x1, max(y1 - 6, 14)), FONT, 0.5, color, 2)
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


def _at_side(det, frame_w: float) -> str:
    side = _side((det.xyxy[0] + det.xyxy[2]) / 2, frame_w)
    return "in the center" if side == "center" else f"on the {side}"


def compose_alert_messages(hazard_pairs, moving_vehicles, elevated, workers,
                           frame_w: float, ppe_violations=None) -> dict[str, str]:
    """Scene-specific spoken phrases built from detection geometry — composed
    in microseconds at the moment of detection, no model involved, so the
    descriptive voice alert still fires on the spot."""
    def at(det):
        return _at_side(det, frame_w)

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
    if ppe_violations:
        n = len(ppe_violations)
        who = "One worker" if n == 1 else f"{n} workers"
        msgs["ppe"] = (f"Warning! {who} without required protective equipment "
                       f"{at(ppe_violations[0])}.")
    return msgs


# --------------------------------------------------------------------------
# Detection loop
# --------------------------------------------------------------------------

def detection_loop(args, detector, in_slot: LatestFrame,
                   out_slot: LatestFrame, state: SessionState, engine: AlertEngine,
                   stop_flag: threading.Event, analyst: IncidentAnalyst | None = None,
                   raw_slot: LatestFrame | None = None, ppe_detector=None):
    motion = None
    frame_diag = None
    frame_shape = None
    last_seq = 0
    fps_smooth = None
    height_since = None       # when the current elevated streak began
    last_height_check = 0.0   # last VLM PPE verification
    last_ppe_check = 0.0      # last full-frame PPE_Detect pass
    ppe_violations: list[Detection] = []  # held over between throttled checks
    had_workers = False       # for an instant PPE check when a worker first appears

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
        # relative worker-vs-worker height check is always on; the absolute
        # upper-zone rule additionally applies with --height-zone
        elevated = find_height_hazards(detections, frame.shape[0],
                                       config.HEIGHT_ZONE_FRACTION,
                                       use_zone=args.height_zone)

        active = set()
        if hazard_pairs:
            active.add("proximity")
        if moving_vehicles and workers:
            active.add("vehicle")
        # Height is verified, not alarmed raw: workers legitimately work at
        # height. A stable candidate goes to the VLM for a PPE check; the
        # alarm only fires if no fall protection is visible (or the check is
        # impossible). Without the analyst, fall back to the raw geometric alarm.
        if elevated:
            if height_since is None:
                height_since = t0
            if analyst is None:
                active.add("height")
            elif (t0 - height_since >= 2.0
                    and t0 - last_height_check >= 30.0):
                last_height_check = t0
                worker = elevated[0]
                msg = (f"Warning! Worker at height {_at_side(worker, frame.shape[1])} "
                       "without visible fall protection. Check fall protection.")
                analyst.submit_height(msg, crop_person(frame, worker))
        else:
            height_since = None

        # Baseline PPE compliance: a second, lightweight YOLO pass over the
        # full frame on its own interval (not every frame — it's a whole
        # extra model). The verdict is held over between checks so it stays
        # stable across the debounce window instead of flickering. A worker
        # newly entering an empty frame forces an immediate check instead of
        # waiting out the interval — someone walking into view should get
        # checked right away, not up to PPE_CHECK_INTERVAL_SEC late (user
        # feedback 2026-07-19: "PPE should check first").
        worker_just_appeared = bool(workers) and not had_workers
        had_workers = bool(workers)
        if ppe_detector is not None and (worker_just_appeared
                or t0 - last_ppe_check >= config.PPE_CHECK_INTERVAL_SEC):
            last_ppe_check = t0
            ppe_dets = ppe_detector.predict(frame, conf=config.PPE_CONF, imgsz=args.imgsz)
            ppe_violations = find_ppe_violations(workers, ppe_dets)
        if ppe_violations and workers:
            active.add("ppe")

        fired = engine.update(
            active,
            detail={"frame": state.frames, "workers": len(workers),
                    "pairs": len(hazard_pairs)},
            messages=compose_alert_messages(hazard_pairs, moving_vehicles,
                                            elevated, workers, frame.shape[1],
                                            ppe_violations=ppe_violations))

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
            display = active | ({"height"}
                                if t0 < state.height_flash_until else set())
            state.active_hazards = sorted(display)
            if hazard_pairs:
                state.proximity_events.append({
                    "frame": state.frames,
                    "timestamp_sec": round(time.time() - state.start_time, 2),
                    "pairs": len(hazard_pairs),
                })
                del state.proximity_events[:-5000]

        annotated = annotate(frame, detections, sorted(display), hazard_pairs,
                            ppe_violations=ppe_violations)
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
                 sources: "SourceManager", raw_slot: LatestFrame | None = None,
                 in_slot: LatestFrame | None = None, tts_voice=None):
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
            elif self.path.startswith("/tts"):
                self._tts()
            elif self.path.startswith("/reset"):
                self._reset()
            elif self.path.startswith("/arm"):
                self._arm()
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path.startswith("/ingest"):
                self._ingest()
                return
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
            state.reset_session()  # a new video is a new session, not a continuation
            print(f"[live] uploaded video now playing: {safe}")
            body = json.dumps({"ok": True, "source": safe}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _ingest(self):
            """Viewer-device camera frames (JPEG per POST). ?claim=1 makes
            the browser the active source; frame posts are rejected with 409
            once another source takes over, which tells the sender to stop."""
            from urllib.parse import parse_qs, urlparse
            if parse_qs(urlparse(self.path).query).get("claim"):
                sources.use_browser()
                print("[live] source switched to: viewer device camera")
                self._static('{"ok": true}', "application/json")
                return
            if sources.current != "browser" or not sources.camera_on:
                self.send_error(409, "browser is not the active source")
                return
            length = int(self.headers.get("Content-Length", 0))
            if not (0 < length <= 3_000_000):
                self.send_error(400)
                return
            data = self.rfile.read(length)
            try:
                import numpy as np
                frame = cv2.imdecode(np.frombuffer(data, np.uint8),
                                     cv2.IMREAD_COLOR)
            except Exception:
                frame = None
            if frame is not None and in_slot is not None:
                in_slot.put(frame)
            self._static('{"ok": true}', "application/json")

        def _static(self, text: str, content_type: str):
            body = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _reset(self):
            """Explicit session reset — called by the dashboard when a tab
            switch stops the previous mode's source. Deliberately a separate
            action from /camera?on=0: the meeting-style auto-off (everyone
            closed their tab, or a manual camera-off toggle) must NOT wipe
            a session's accumulated stats just because the same source will
            resume in a few seconds — only an actual mode change should."""
            state.reset_session()
            self._static('{"ok": true}', "application/json")

        def _camera(self):
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            if "on" in qs:
                on = qs["on"][0] == "1"
                sources.set_camera(on)
                if not on:
                    with state.lock:  # no camera, no live hazards
                        state.active_hazards = []
                        state.workers_now = state.vehicles_now = 0
                    def push_off():
                        if sources.camera_on:
                            return
                        placeholder = camera_off_frame()
                        if raw_slot is not None:
                            raw_slot.put(placeholder.copy())
                        out_slot.put(placeholder)
                    push_off()
                    # the dying capture thread may land one final frame after
                    # our placeholder — repaint it once things settle
                    threading.Timer(0.8, push_off).start()
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
                state.reset_session()  # a different source is a new session
                print(f"[live] source switched to: {raw}")
            body = json.dumps({"ok": bool(raw),
                               "source": public_source(sources.current)}).encode()
            self.send_response(200 if raw else 400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _arm(self):
            """Point SourceManager.current at a device WITHOUT opening it —
            consent model: the camera only actually opens once the user
            presses the camera button. Exists to fix a real bug (2026-07-19):
            entering the Camera tab used to only call /camera?on=0, which
            stops whatever was running but leaves `current` exactly where it
            was — still an uploaded video's file path, or the raw startup
            default index (often the wrong node on a multi-stream camera).
            Pressing the camera-on button afterward would then resume that
            stale/wrong source instead of the dropdown's recommended camera.
            One decisive call instead of two also removes a race: the old
            two-call "stop, then let the user separately pick" sequence
            could have its own stop response land AFTER a fast follow-up
            camera selection and turn it back off."""
            from urllib.parse import parse_qs, urlparse
            raw = (parse_qs(urlparse(self.path).query).get("src") or [""])[0].strip()
            if raw:
                sources.arm(parse_source(raw))
                state.reset_session()
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
            from urllib.parse import parse_qs, urlparse
            fmt = (parse_qs(urlparse(self.path).query).get("format") or [""])[0]
            if fmt == "json":
                body = json.dumps(state.to_report_dict(), indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition",
                                 'attachment; filename="hazard_report.json"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            try:
                from report_generation import generate_hazard_report
                text = generate_hazard_report(state.to_report_dict())
                status = 200
            except Exception as exc:
                text, status = f"Report generation failed: {exc}", 500
            if fmt == "pdf" and status == 200:
                from report_generation import report_to_pdf
                body = report_to_pdf(text)
                self.send_response(200)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Disposition",
                                 'attachment; filename="site_safety_report.pdf"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            body = text.encode()
            self.send_response(status)
            if fmt == "txt" and status == 200:
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Disposition",
                                 'attachment; filename="site_safety_report.txt"')
            else:
                self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _tts(self):
            """Server-synthesized alert audio (Piper) — see load_tts_voice().
            404 tells the dashboard JS to fall back to the browser's own
            Web Speech API for this utterance; never blocks or 500s over a
            speech engine hiccup, voice alerts must never crash the page."""
            if tts_voice is None:
                self.send_error(404, "server TTS not available")
                return
            from urllib.parse import parse_qs, urlparse
            text = (parse_qs(urlparse(self.path).query).get("text") or [""])[0][:500]
            if not text:
                self.send_error(400, "missing text")
                return
            try:
                import io
                import wave
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wav_file:
                    tts_voice.synthesize_wav(text, wav_file)
                body = buf.getvalue()
            except Exception as exc:
                self.send_error(500, f"synthesis failed: {exc}")
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Cache-Control", "no-store")
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
        if viewers == 0 and auto and sources.camera_on and not sources.is_file_source:
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
a.dlbtn{background:transparent;border:1px solid var(--cyan);color:var(--deep);
padding:8px 14px;border-radius:2px;cursor:pointer;font-size:12px;
letter-spacing:.12em;text-transform:uppercase;font-weight:600;
font-family:"Barlow Condensed","Segoe UI",sans-serif;text-decoration:none;
display:none}
a.dlbtn:hover{background:var(--tint);border-color:var(--cyandark)}
a.dlbtn.show{display:inline-block}
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
#viewwrap.blanked #view{visibility:hidden}
#viewBlank{display:none;position:absolute;inset:0;z-index:2;
align-items:center;justify-content:center;text-align:center;padding:0 40px;
color:rgba(255,255,255,.55);font-size:13px;letter-spacing:.04em;
background:#0d1418}
#viewwrap.blanked #viewBlank{display:flex}
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
.dd{position:relative;cursor:pointer;user-select:none;padding-right:12px}
.ddlabel{padding:9px 0;max-width:230px;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap}
.dd .chev{width:13px;height:13px}
.menu{position:absolute;top:calc(100% + 6px);left:0;min-width:100%;
background:#fff;border:1px solid var(--line);border-radius:12px;
box-shadow:0 10px 28px rgba(18,42,51,.14);list-style:none;padding:6px;
display:none;z-index:30;max-height:230px;overflow-y:auto;white-space:nowrap}
.dd.open .menu{display:block}
.dd.open{border-color:var(--cyan)}
.menu li{padding:9px 13px;border-radius:7px;font-size:12.5px;color:var(--ink)}
.menu li:hover{background:var(--tint)}
.menu li.sel{color:var(--deep);font-weight:600;background:var(--tint)}
.menu li.none{color:var(--mut);cursor:default}
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
button.pill.on{border-color:var(--cyan);background:var(--tint);color:var(--deep)}
.circ svg{width:22px;height:22px;fill:none;stroke:#fff;stroke-width:2;
stroke-linecap:round;stroke-linejoin:round}
#srcnow{font-size:11px;color:var(--mut);width:100%;text-align:center}
#srctabs{display:flex;gap:8px;width:100%;justify-content:center}
.tabbtn{font-weight:600;letter-spacing:.02em}
.tabpanel{display:flex;gap:10px;flex-wrap:wrap;align-items:center;justify-content:center;
width:100%}
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
      <div id="viewBlank">Pick a camera or upload a video to begin</div>
      <span id="namechip">Site camera</span>
      <span id="viewtag">ANNOTATED</span>
      <div id="vidctl">
        <button class="circ" id="cambtn" title="Turn camera on/off"></button>
        <button class="circ" id="flipbtn" title="Mirror / flip camera"></button>
        <button class="circ" id="mutebtn" title="Voice alerts on/off"></button>
      </div>
    </div>
    <div id="pills">
      <div id="srctabs">
        <button class="pill tabbtn on" id="tab-camera" data-tab="camera">Live Camera</button>
        <button class="pill tabbtn" id="tab-upload" data-tab="upload">Upload Video</button>
      </div>
      <div class="tabpanel" id="panel-camera">
        <div class="pillwrap dd" id="camdd" title="Camera device">
          <svg viewBox="0 0 24 24"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
          <span class="ddlabel" id="camlabel">scanning cameras…</span>
          <svg class="chev" viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>
          <ul class="menu" id="cammenu"></ul>
        </div>
        <button class="iconbtn" id="rescan" title="Rescan for new cameras">
          <svg viewBox="0 0 24 24"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
        </button>
        <button class="pill" id="devcam" title="Stream this device's camera to the server for detection">
          <svg viewBox="0 0 24 24"><rect x="5" y="2" width="14" height="20" rx="2"/><circle cx="12" cy="11" r="3.2"/></svg>
          Use this device camera</button>
      </div>
      <div class="tabpanel" id="panel-upload" style="display:none">
        <button class="pill" id="upbtn" title="Run detection on a video file (loops)">
          <svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          Upload video</button>
        <input type="file" id="upfile" accept="video/*" hidden>
        <button class="pill" id="stopvidbtn" title="Stop the uploaded video and return to standby">
          <svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="1"/></svg>
          Stop video</button>
      </div>
      <div class="pillwrap dd" id="viewdd" title="View">
        <svg viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
        <span class="ddlabel" id="viewlabel">Annotated view</span>
        <svg class="chev" viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"/></svg>
        <ul class="menu" id="viewmenu">
          <li data-v="/stream.mjpg" data-tag="ANNOTATED" class="sel">Annotated view</li>
          <li data-v="/raw.mjpg" data-tag="DIRECT">Direct view</li>
        </ul>
      </div>
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
      <div class="hzrow" id="hz-ppe">
        <div><div class="k">PPE</div><div class="d">Worker missing hardhat or safety vest</div></div>
        <span class="n" id="n-ppe">0</span></div>
    </div>
    <h2>Alert log</h2>
    <ul id="log"><li>No alerts yet.</li></ul>
    <h2>On-the-spot AI analysis</h2>
    <ul id="incidents"><li>Scene notes appear here seconds after an alert fires.</li></ul>
    <h2>Safety report</h2>
    <div class="foot">
      <button id="reportbtn">Generate report</button>
      <a id="dlpdf" class="dlbtn" href="/report?format=pdf">Download PDF</a>
      <a id="dltxt" class="dlbtn" href="/report?format=txt">Download TXT</a>
      <a id="dljson" class="dlbtn" href="/report?format=json">Download JSON</a>
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
let serverMode='camera',modeSynced=false;
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
let voiceGen=0;
function stopVoiceNow(){ // flush the queue + cancel whichever engine is speaking
  voiceGen++;  // a /tts fetch already in flight must not start playing after this
  speech.q=[];speech.busy=false;
  if(window.speechSynthesis)speechSynthesis.cancel();
  if(currentAudio){currentAudio.pause();currentAudio=null;}}
$('mutebtn').onclick=()=>{muted=!muted;
  $('mutebtn').innerHTML=muted?IC.volOff:IC.vol;
  $('mutebtn').classList.toggle('off',muted);
  if(muted)stopVoiceNow();
  if(!muted)say('Voice alerts enabled',{pri:3,ttl:3000});};
// Voice pick: browsers load voices asynchronously and often default to a
// low-quality/robotic one (espeak on bare Linux) even when better ones are
// installed. Prefer a named "premium"/neural voice, then any local English
// voice, then whatever the browser hands us — never block on this.
let bestVoice=null;
function pickVoice(){
  if(!window.speechSynthesis)return;
  const voices=speechSynthesis.getVoices();
  if(!voices.length)return;
  const scored=voices.map(v=>{
    const n=v.name.toLowerCase();
    let score=0;
    if(/natural|neural|premium|enhanced|siri|google/.test(n))score+=3;
    if(v.localService)score+=1;
    if(v.lang&&v.lang.toLowerCase().startsWith('en'))score+=1;
    if(/espeak|mbrola|generic/.test(n))score-=3;
    return {v,score};
  });
  scored.sort((a,b)=>b.score-a.score);
  bestVoice=scored[0].v;
}
if(window.speechSynthesis){
  pickVoice();
  speechSynthesis.onvoiceschanged=pickVoice;
}
// Priority speech queue: an utterance always finishes its sentence; the
// next one waits its turn; anything that sat unspoken past its ttl is
// dropped (a stale alarm is noise, not information). Elegant "on the spot"
// fix (2026-07-19): rather than speaking a backlog one utterance at a time
// (which reads as delayed/queued once more than one alert is waiting),
// pumpSpeech drains and combines EVERY currently-queued item at the top
// priority into one sentence each time it's ready to speak — a burst of
// alerts becomes one slightly longer sentence, not a growing queue.
const speech={q:[],busy:false};
let currentAudio=null;
let ttsAvailable=true;  // optimistic; flips false permanently on first /tts failure
function say(text,opts){
  if(muted)return;
  const o=opts||{};
  speech.q.push({text,pri:o.pri||2,exp:Date.now()+(o.ttl||6000)});
  pumpSpeech();}
function stripPrefix(t){return t.replace(/^(Warning!|Caution!)\\s*/,'');}
async function pumpSpeech(){
  if(speech.busy)return;
  const now=Date.now();
  speech.q=speech.q.filter(i=>i.exp>now);
  if(!speech.q.length)return;
  speech.q.sort((a,b)=>b.pri-a.pri);
  const topPri=speech.q[0].pri;
  const batch=[];
  while(speech.q.length&&speech.q[0].pri===topPri)batch.push(speech.q.shift());
  const text=batch.length===1?batch[0].text:
    batch.length+' hazards. '+batch.map(b=>stripPrefix(b.text)).join(' Also, ');
  speech.busy=true;
  const myGen=voiceGen;  // if stopVoiceNow() runs while we're mid-fetch (e.g. a tab
  const stale=()=>myGen!==voiceGen;  // switch), this utterance must not start playing after
  const next=()=>{speech.busy=false;currentAudio=null;setTimeout(pumpSpeech,400);};
  if(ttsAvailable){
    try{
      const r=await fetch('/tts?text='+encodeURIComponent(text));
      if(stale()){speech.busy=false;return;}
      if(!r.ok)throw new Error('no server voice');
      const blob=await r.blob();
      if(stale()){speech.busy=false;return;}
      const audio=new Audio(URL.createObjectURL(blob));
      currentAudio=audio;
      audio.onended=audio.onerror=next;
      await audio.play();
      return;
    }catch(e){if(stale()){speech.busy=false;return;} ttsAvailable=false;}
    // this session: always use the browser voice from here on (genuine failure, not a stale fetch)
  }
  if(stale()){speech.busy=false;return;}
  if(!window.speechSynthesis){speech.busy=false;return;}
  const u=new SpeechSynthesisUtterance(text);
  if(bestVoice)u.voice=bestVoice;
  u.rate=1.0;u.pitch=1.0;u.volume=1;
  u.onend=u.onerror=next;
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
    for(const k of['proximity','vehicle','height','ppe']){
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
      const fresh=d.recent_alerts.filter(a=>a.time>lastAlert);
      if(fresh.length){
        if(lastAlert>0) // don't replay history on page load — pumpSpeech
          fresh.forEach(a=>say(a.message,{pri:2}));   // batches any backlog itself
        lastAlert=fresh[fresh.length-1].time;}
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
        say('Detail: '+nw.note.slice(0,220),{pri:1,ttl:20000});
      lastIncKey=key;
    }
    if(d.source!==undefined){
      const digit=/^[0-9]+$/.test(d.source);
      const cam=digit?camList.find(c=>String(c.index)===d.source):null;
      const label=cam?cam.label:(digit?'Camera '+d.source
        :(d.source==='browser'?'Device camera (streamed)'
          :d.source.split(/[\\/]/).pop()));
      $('namechip').textContent=(d.camera_on?'':'OFF · ')+label;
      if(digit&&camSelected!==d.source&&!$('camdd').classList.contains('open')){
        camSelected=d.source;renderCamMenu();}
      if(!$('srcnow').textContent.startsWith('Switching'))
        $('srcnow').textContent=(d.camera_on?'live':'camera off')
          +'  ·  viewers: '+d.viewers;
      // Ground truth for "is a tab click actually changing anything" — a
      // client-only guess (e.g. a variable defaulting to 'camera' on every
      // fresh page load) goes stale the moment a video's been left running
      // from an earlier session, so clicking the already-highlighted Live
      // Camera tab silently did nothing. This tracks the server's own idea
      // of the mode instead. Also syncs which tab is VISIBLE the first time
      // (page load / refresh) so a page opened onto a playing upload shows
      // the Upload tab, not a stale default.
      const trueMode=(digit||d.source==='browser')?'camera':'upload';
      if(!modeSynced){selectTab(trueMode);modeSynced=true;}
      serverMode=trueMode;}
    if(d.camera_on!==undefined){
      camOn=d.camera_on;
      $('cambtn').innerHTML=camOn?IC.cam:IC.camOff;
      $('cambtn').classList.toggle('off',!camOn);
      // Server truth, checked every poll — covers the tab-switch instant
      // blank AND auto-off AND a manual toggle with one rule, rather than
      // waiting for the MJPEG stream to eventually deliver a placeholder
      // frame (up to ~1s lag) to visually reflect "nothing is active".
      $('viewwrap').classList.toggle('blanked',!camOn);
      if(!camOn){
        // Distinguish "a camera is armed, one more click starts it" from
        // "nothing picked yet" — a generic message in both cases (2026-07-19
        // user confusion) left it unclear that the recommended camera was
        // already selected and just needed the camera button pressed.
        const armedCam=camList.find(c=>String(c.index)===camSelected&&c.usable!==false);
        $('viewBlank').textContent=armedCam
          ?'Camera ready ('+armedCam.label.split(' — ')[0]+') — press the camera button below to start'
          :'Pick a camera or upload a video to begin';}
      maybeAutoArmOnLoad();}
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
let camSelected='';
function renderCamMenu(){
  const menu=$('cammenu');menu.innerHTML='';
  camList.forEach(c=>{const li=document.createElement('li');
    li.dataset.i=String(c.index);li.textContent=c.label;
    // Labeling a bad entry "not usable" wasn't foolproof enough on its own
    // (2026-07-19) — reuse the same "none" class the empty-list placeholder
    // already uses, which the click handler below already ignores, so a
    // known-non-color node genuinely can't be selected, not just discouraged.
    if(c.usable===false)li.classList.add('none');
    if(String(c.index)===camSelected)li.classList.add('sel');
    menu.appendChild(li);});
  if(!camList.length){const li=document.createElement('li');
    li.className='none';li.textContent='no camera found — plug one in';
    menu.appendChild(li);}
  const cur=camList.find(c=>String(c.index)===camSelected);
  if(cur)$('camlabel').textContent=cur.label;
  else if(!camList.length)$('camlabel').textContent='no camera found';
  // Bug (2026-07-19): with cameras found but none picked yet, neither branch
  // above matched, so the label stayed stuck on its initial HTML placeholder
  // "scanning cameras…" forever — looked like it was perpetually scanning
  // even though the scan had long since finished and options were ready.
  else $('camlabel').textContent=camList.length+' camera'+(camList.length>1?'s':'')+' found — pick one';}
// Arms (points `current` at, without opening) the first usable=true camera
// — the same one the dropdown already sorts to the top and marks
// "(recommended)". Returns true if one existed to arm. Used both when
// entering the Camera tab and once on initial page load, so a fresh server
// (armed to whatever raw index --source defaulted to, often wrong for a
// multi-stream camera) starts pointed at the right device before the user
// ever presses camera-on, not after.
async function armRecommendedCamera(){
  const usable=camList.filter(c=>c.usable!==false);
  if(!usable.length)return false;
  const rec=usable[0];
  camSelected=String(rec.index);renderCamMenu();
  try{await fetch('/arm?src='+encodeURIComponent(rec.index));}catch(e){}
  return true;
}
let autoArmed=false;
function maybeAutoArmOnLoad(){
  // Only touch things while genuinely idle: camera mode, nothing already
  // streaming, and not already pointed at a usable camera (don't fight a
  // deliberate earlier choice this session).
  if(autoArmed||!modeSynced||serverMode!=='camera'||camOn)return;
  const usable=camList.filter(c=>c.usable!==false);
  if(!usable.length)return;
  if(usable.some(c=>String(c.index)===camSelected)){autoArmed=true;return;}
  autoArmed=true;
  armRecommendedCamera();
}
async function loadCams(refresh){
  if(refresh)$('rescan').classList.add('spin');
  try{const r=await fetch('/cameras'+(refresh?'?refresh=1':''));
    const fresh=await r.json();
    if(camList.length){
      const known=new Set(camList.map(c=>c.index));
      fresh.filter(c=>!known.has(c.index)).forEach(c=>{
        $('srcnow').textContent='New camera detected: '+c.label;});}
    camList=fresh;renderCamMenu();
    maybeAutoArmOnLoad();
  }catch(e){}
  $('rescan').classList.remove('spin');}
loadCams();
// auto-detect newly plugged cameras — name-list only, never opens a device
setInterval(()=>loadCams(false),20000);
$('rescan').onclick=e=>{e.stopPropagation();loadCams(true);};
// custom dropdowns (native <select> popups can't be styled)
document.querySelectorAll('.dd').forEach(el=>{
  el.addEventListener('click',e=>{
    const li=e.target.closest('li');
    if(li&&!li.classList.contains('none')){
      el.classList.remove('open');
      if(el.id==='camdd'){camSelected=li.dataset.i;renderCamMenu();
        switchSrc(li.dataset.i);}
      else{$('view').src=li.dataset.v;$('viewtag').textContent=li.dataset.tag;
        $('viewlabel').textContent=li.textContent;
        el.querySelectorAll('li').forEach(x=>x.classList.remove('sel'));
        li.classList.add('sel');}
      return;}
    document.querySelectorAll('.dd.open').forEach(d=>{if(d!==el)d.classList.remove('open');});
    el.classList.toggle('open');});});
document.addEventListener('click',e=>{
  if(!e.target.closest('.dd'))
    document.querySelectorAll('.dd.open').forEach(d=>d.classList.remove('open'));});
// Viewer-device camera -> server: getUserMedia frames posted as JPEG.
// Needs a secure context (HTTPS or localhost) — hidden otherwise.
let devStream=null,devTimer=null;
if(!(window.isSecureContext&&navigator.mediaDevices
     &&navigator.mediaDevices.getUserMedia))
  $('devcam').style.display='none';
function stopDevCam(){
  if(devTimer){clearInterval(devTimer);devTimer=null;}
  if(devStream){devStream.getTracks().forEach(t=>t.stop());devStream=null;}
  $('devcam').classList.remove('on');}
async function startDevCam(){
  try{devStream=await navigator.mediaDevices.getUserMedia(
    {video:{width:{ideal:640},facingMode:'environment'},audio:false});}
  catch(e){$('srcnow').textContent='Camera permission denied.';return;}
  await fetch('/ingest?claim=1',{method:'POST'});
  const v=document.createElement('video');
  v.srcObject=devStream;v.muted=true;v.playsInline=true;await v.play();
  const cv=document.createElement('canvas');
  let busy=false;
  devTimer=setInterval(()=>{
    if(busy||!v.videoWidth)return;
    cv.width=v.videoWidth;cv.height=v.videoHeight;
    cv.getContext('2d').drawImage(v,0,0);
    cv.toBlob(async b=>{
      if(!b)return;busy=true;
      try{const r=await fetch('/ingest',{method:'POST',body:b});
        if(r.status===409)stopDevCam();} // another source took over
      catch(e){}
      busy=false;},'image/jpeg',0.7);},170);
  $('devcam').classList.add('on');
  $('srcnow').textContent='Streaming this device camera to the server.';}
$('devcam').onclick=()=>devStream?stopDevCam():startDevCam();
window.addEventListener('pagehide',stopDevCam);
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
  $('upfile').value='';
  selectTab('upload');};
$('stopvidbtn').onclick=async()=>{
  try{await fetch('/camera?on=0');
    $('srcnow').textContent='Video stopped — pick a camera or upload another clip.';}
  catch(e){}};
// Camera and Upload are two distinct modes. Switching tabs stops whatever
// the mode you're LEAVING was actually doing on the SERVER (2026-07-19:
// leaving an uploaded video running in the background after switching to
// Live Camera was confusing — "why can I still hear it" — so the switch
// itself tears it down). Compared against serverMode (synced from /events
// every poll), not a client-only guess — a plain click-tracked variable
// defaults to 'camera' on every fresh page load even when a video is
// already playing from an earlier session, so clicking the
// already-highlighted Live Camera tab would silently do nothing.
function selectTab(name){
  document.querySelectorAll('.tabbtn').forEach(b=>
    b.classList.toggle('on',b.dataset.tab===name));
  $('panel-camera').style.display=name==='camera'?'flex':'none';
  $('panel-upload').style.display=name==='upload'?'flex':'none';
}
document.querySelectorAll('.tabbtn').forEach(btn=>
  btn.onclick=async()=>{
    const target=btn.dataset.tab;
    selectTab(target);
    if(serverMode!==target){
      stopDevCam();
      stopVoiceNow();           // an alert from the mode you're leaving must not keep talking
      $('viewwrap').classList.add('blanked');  // instant — don't wait for the server's
      $('view').alt='';                        // placeholder frame to arrive over MJPEG
      // New mode, new session — old counts/log/report from a different
      // video or camera would otherwise sit there looking current.
      $('log').innerHTML='<li>No alerts yet.</li>';
      $('incidents').innerHTML='<li>Scene notes appear here seconds after an alert fires.</li>';
      $('livereport').style.display='none';$('livereport').textContent='';
      $('banner').classList.remove('on');
      for(const k of['proximity','vehicle','height','ppe']){
        $('n-'+k).textContent='0';$('hz-'+k).classList.remove('active');}
      $('alerts').textContent='0';$('workers').textContent='0';$('vehicles').textContent='0';
      lastAlert=Date.now()/1000;lastIncKey=null;
      // Entering Camera: arm the recommended camera in ONE decisive call
      // instead of a bare /camera?on=0 (2026-07-19 bugs, both from the same
      // root cause — /camera?on=0 stops whatever was running but leaves
      // `current` untouched): (1) pressing camera-on afterward used to
      // resume the stale uploaded video, or the raw startup default index
      // (often the wrong node on a multi-stream camera like RealSense), not
      // the dropdown's recommended camera; (2) selecting a camera right
      // after switching could race this call's own async response, which
      // could land after and turn the just-started camera back off. /arm
      // both fixes `current` and removes the second stop-call entirely.
      if(target==='camera' && await armRecommendedCamera()){/* armed */}
      else{try{await Promise.all([fetch('/camera?on=0'),fetch('/reset')]);}catch(e){}}
      $('srcnow').textContent='';
      serverMode=target;
    }});
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
  try{
    const r=await fetch('/report');box.textContent=await r.text();
    if(r.ok){['dlpdf','dltxt','dljson'].forEach(id=>$(id).classList.add('show'));}
  }catch(e){box.textContent='Report request failed: '+e;}};
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
    p.add_argument("--audio", action="store_true",
                   help="Enable server-side speaker output (e.g. a speaker on "
                        "the Spark). Default off — the dashboard's browser "
                        "voice speaks the descriptive alert on the spot.")
    p.add_argument("--no-audio", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no-analysis", action="store_true",
                   help="Disable on-the-spot VLM/LLM incident analysis")
    p.add_argument("--autostart", action="store_true",
                   help="Open the camera immediately (unattended/server mode). "
                        "Default: start in standby — the camera stays off until "
                        "someone presses the camera button on the dashboard.")
    p.add_argument("--height-zone", action="store_true",
                   default=config.HEIGHT_ZONE_ENABLED,
                   help="Also enable the absolute upper-zone height rule "
                        "(the relative worker-vs-worker check is always on)")
    p.add_argument("--no-ppe", action="store_true",
                   help="Disable the PPE_Detect model: no baseline PPE-"
                        "compliance hazard, and height alarms always fire "
                        "(the hardhat cross-check can never pass)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mock and args.device == "0":
        args.device = "cpu"

    detector = build_detector(args)
    print(f"[live] detector: {getattr(detector, 'label', 'Plan A')}")
    print(f"[live] Ollama (Spark): {config.OLLAMA_HOST}")

    ppe_detector = None
    if config.PPE_ENABLED and not args.no_ppe:
        if config.PPE_MODEL_PATH.exists():
            from detectors import PPEDetector
            ppe_detector = PPEDetector(str(config.PPE_MODEL_PATH), device=args.device).load()
            print(f"[live] PPE detector: {config.PPE_MODEL_PATH}")
        else:
            print(f"[live] PPE detector disabled — weights not found at "
                  f"{config.PPE_MODEL_PATH} (no baseline PPE hazard; height "
                  "alarms will always fire, never suppressed)")

    tts_voice = load_tts_voice()

    player = AudioPlayer() if (args.audio and not args.no_audio) else None
    engine = AlertEngine(player=player)
    state = SessionState(engine)
    state.detector_label = getattr(detector, "label", "Plan A — Fine-tuned YOLOv8")

    source = args.video if args.video else parse_source(args.source)
    in_slot, out_slot, raw_slot = LatestFrame(), LatestFrame(), LatestFrame()
    sources = SourceManager(in_slot)
    if args.video or args.autostart:
        sources.start(source, loop_file=bool(args.video), critical=True)
    else:
        # meeting-style standby: nothing is captured until a user consents by
        # pressing the camera button on the dashboard
        sources.arm(source)
        standby = camera_off_frame("CAMERA OFF - press the camera button")
        raw_slot.put(standby.copy())
        out_slot.put(standby)
        print("[live] standby: camera stays OFF until turned on from the dashboard")

    threading.Thread(target=spark_status_poller, args=(state,), daemon=True).start()

    analyst = None
    if not args.no_analysis:
        analyst = IncidentAnalyst(state, engine, ppe_detector=ppe_detector)
        analyst.start()

    stop_flag = threading.Event()
    det_thread = threading.Thread(
        target=detection_loop,
        args=(args, detector, in_slot, out_slot, state, engine, stop_flag, analyst,
              raw_slot, ppe_detector),
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
                                                      raw_slot, in_slot, tts_voice))
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
