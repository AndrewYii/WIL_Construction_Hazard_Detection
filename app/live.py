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

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[capture] ERROR: cannot open source {self.source!r}")
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
        self.slot.close()


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
            }


def spark_status_poller(state: SessionState, interval: float = 30.0):
    """Background refresh of the Spark Ollama status for the dashboard."""
    from llm_client import get_client
    while True:
        try:
            status = get_client().status()
        except Exception:
            status = {"up": False, "host": config.OLLAMA_HOST, "models": []}
        with state.lock:
            state.spark_status = status
        time.sleep(interval)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

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
    return frame


def draw_hud(frame, state: SessionState):
    hud = f"{state.fps:.1f} FPS | workers {state.workers_now} | vehicles {state.vehicles_now}"
    cv2.putText(frame, hud, (12, 26), FONT, 0.65, (0, 0, 0), 4)
    cv2.putText(frame, hud, (12, 26), FONT, 0.65, (255, 255, 255), 2)
    return frame


# --------------------------------------------------------------------------
# Detection loop
# --------------------------------------------------------------------------

def detection_loop(args, detector, in_slot: LatestFrame,
                   out_slot: LatestFrame, state: SessionState, engine: AlertEngine,
                   stop_flag: threading.Event):
    motion = None
    frame_diag = None
    last_seq = 0
    fps_smooth = None

    while not stop_flag.is_set():
        frame, last_seq = in_slot.get(last_seq)
        if frame is None:
            if in_slot.closed:
                break
            continue
        t0 = time.time()

        if frame_diag is None:
            h, w = frame.shape[:2]
            frame_diag = math.hypot(w, h)
            motion = VehicleMotionTracker(frame_diag, config.VEHICLE_MOVE_RATIO_PER_SEC)

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

        engine.update(active, detail={"frame": state.frames,
                                      "workers": len(workers),
                                      "pairs": len(hazard_pairs)})

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

    out_slot.close()


# --------------------------------------------------------------------------
# HTTP server (headless mode)
# --------------------------------------------------------------------------

def make_handler(out_slot: LatestFrame, state: SessionState):
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
            elif self.path.startswith("/stream"):
                self._stream()
            elif self.path.startswith("/events"):
                body = json.dumps(state.snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/report"):
                self._report()
            else:
                self.send_error(404)

        def _stream(self):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            seq = 0
            params = [cv2.IMWRITE_JPEG_QUALITY, config.MJPEG_QUALITY]
            try:
                while True:
                    frame, seq = out_slot.get(seq)
                    if frame is None:
                        if out_slot.closed:
                            break
                        continue
                    # JPEG encoding happens here, in the serving thread,
                    # never in the detection loop
                    ok, jpg = cv2.imencode(".jpg", frame, params)
                    if not ok:
                        continue
                    self.wfile.write(b"--frame\r\n"
                                     b"Content-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpg)).encode()
                                     + b"\r\n\r\n")
                    self.wfile.write(jpg.tobytes())
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # client closed the tab

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
<style>
:root{--bg:#0d1418;--panel:#132027;--line:#1f3540;--ink:#e8f1f4;--mut:#7fa0ac;
--cyan:#22b8cf;--red:#ff4d4d;--amber:#ffb020;--green:#3ddc84;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font:14px/1.45 "Segoe UI",system-ui,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:10px 18px;
border-bottom:2px solid var(--cyan);background:var(--panel)}
header .tag{background:var(--cyan);color:#04252b;font-weight:700;letter-spacing:.18em;
padding:3px 10px;font-size:11px;text-transform:uppercase;border-radius:2px}
header h1{font-size:15px;letter-spacing:.12em;text-transform:uppercase;font-weight:600}
header .right{margin-left:auto;display:flex;gap:10px;align-items:center}
.dot{width:9px;height:9px;border-radius:50%;background:var(--red);display:inline-block}
.dot.ok{background:var(--green)}
main{display:grid;grid-template-columns:minmax(0,2.2fr) minmax(280px,1fr);
gap:14px;padding:14px 18px;max-width:1500px;margin:0 auto}
@media(max-width:900px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:4px;overflow:hidden}
.panel h2{font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--mut);
padding:9px 12px;border-bottom:1px solid var(--line)}
#viewwrap{position:relative}
#view{width:100%;display:block;background:#000;min-height:280px}
#banner{position:absolute;left:0;right:0;top:0;background:rgba(200,20,20,.92);
color:#fff;text-align:center;font-weight:700;letter-spacing:.14em;padding:10px;
font-size:16px;display:none;text-transform:uppercase}
#banner.on{display:block;animation:blink 1s steps(2) infinite}
@keyframes blink{50%{background:rgba(120,0,0,.92)}}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line)}
.stat{background:var(--panel);padding:10px 12px}
.stat .v{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}
.stat .l{font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--mut)}
.hz{display:flex;flex-direction:column;gap:8px;padding:12px}
.hzrow{display:flex;align-items:center;gap:10px;padding:8px 10px;border:1px solid var(--line);
border-radius:3px;background:#0f1a20}
.hzrow.active{border-color:var(--red);background:#2a1214}
.hzrow .n{margin-left:auto;font-weight:700;font-variant-numeric:tabular-nums}
.hzrow .k{font-size:12px;text-transform:uppercase;letter-spacing:.08em}
.hzrow .d{font-size:11px;color:var(--mut)}
#log{list-style:none;max-height:260px;overflow-y:auto;padding:6px 12px;font-size:12px}
#log li{padding:5px 0;border-bottom:1px dashed var(--line);color:var(--mut)}
#log li b{color:var(--red)}
.foot{display:flex;gap:10px;align-items:center;padding:10px 12px;flex-wrap:wrap}
button{background:transparent;border:1px solid var(--cyan);color:var(--cyan);
padding:7px 14px;border-radius:3px;cursor:pointer;font-size:12px;
letter-spacing:.1em;text-transform:uppercase}
button:hover{background:rgba(34,184,207,.12)}
button.mute.off{border-color:var(--mut);color:var(--mut)}
#spark{font-size:11px;color:var(--mut)}
#report{white-space:pre-wrap;font:12px/1.5 Consolas,monospace;padding:12px;display:none;
max-height:340px;overflow-y:auto;border-top:1px solid var(--line)}
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
    <div class="foot">
      <button class="mute" id="mutebtn">🔊 Voice on</button>
      <button id="reportbtn">Generate report</button>
      <span id="spark">Spark: checking…</span>
    </div>
    <div id="report"></div>
  </section>
</main>
<script>
let muted=false,lastAlert=0,alertTotal=0;
const $=id=>document.getElementById(id);
$('mutebtn').onclick=()=>{muted=!muted;
  $('mutebtn').textContent=muted?'🔇 Voice off':'🔊 Voice on';
  $('mutebtn').classList.toggle('off',muted);
  if(!muted)say('Voice alerts enabled');};
function say(text){
  if(muted||!window.speechSynthesis)return;
  const u=new SpeechSynthesisUtterance(text);
  u.rate=1.05;u.pitch=1;u.volume=1;
  speechSynthesis.cancel();speechSynthesis.speak(u);}
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
    const s=d.spark;
    $('spark').textContent=s.up
      ?'Spark LLM: '+(s.report_model||'connected')+' @ '+s.host
      :'Spark LLM: offline ('+s.host+') — reports use template fallback';
  }catch(e){$('livedot').classList.remove('ok');}
}
$('reportbtn').onclick=async()=>{
  const box=$('report');box.style.display='block';
  box.textContent='Generating safety report on the Spark…';
  try{const r=await fetch('/report');box.textContent=await r.text();}
  catch(e){box.textContent='Report request failed: '+e;}};
setInterval(poll,1000);poll();
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

    source = args.video if args.video else (
        int(args.source) if args.source.isdigit() else args.source)
    in_slot, out_slot = LatestFrame(), LatestFrame()
    capture = CaptureThread(source, in_slot, loop_file=bool(args.video))
    capture.start()

    threading.Thread(target=spark_status_poller, args=(state,), daemon=True).start()

    stop_flag = threading.Event()
    det_thread = threading.Thread(
        target=detection_loop,
        args=(args, detector, in_slot, out_slot, state, engine, stop_flag),
        daemon=True)
    det_thread.start()

    server = None
    try:
        if args.headless:
            server = ThreadingHTTPServer(("0.0.0.0", args.port),
                                         make_handler(out_slot, state))
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
        capture.stop_flag.set()
        if server:
            server.shutdown()
        print(f"\n[live] session summary: {json.dumps(state.snapshot()['alert_counts'])}")
        print(f"[live] events log: {config.EVENTS_LOG}")


if __name__ == "__main__":
    main()
