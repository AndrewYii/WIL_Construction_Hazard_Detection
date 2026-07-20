"""
Central configuration for the hazard detection stack.

Everything here can be overridden with environment variables so the same
code runs on the DGX Spark (headless server) and a dev laptop without edits.

    OLLAMA_HOST / SPARK_OLLAMA_HOST  where the Ollama server lives
                                     (Spark default port 11434)
    REPORT_MODELS                    comma-separated preference chain
    VLM_MODELS                       comma-separated preference chain
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# .env lives at the project root (gitignored) and never overrides variables
# already set in the shell, so `run_session.sh`'s own export still wins.
load_dotenv(PROJECT_ROOT / ".env")

# --- Ollama server (NVIDIA DGX Spark) ------------------------------------
# Set OLLAMA_HOST (or SPARK_OLLAMA_HOST) to the Spark's address, e.g.
#   OLLAMA_HOST=http://192.168.1.50:11434
# The port matters too — Ollama's default is 11434, but set it in .env if
# your Spark's instance listens elsewhere. The placeholder below assumes
# the hostname "spark" resolves on your LAN.
OLLAMA_HOST = (
    os.environ.get("SPARK_OLLAMA_HOST")
    or os.environ.get("OLLAMA_HOST")
    or "http://spark:11434"
)
if not OLLAMA_HOST.startswith(("http://", "https://")):
    OLLAMA_HOST = f"http://{OLLAMA_HOST}"

OLLAMA_TIMEOUT_SEC = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "120"))

# --- Model preference chains (first available on the server wins) --------
# Pull on the Spark with:  ollama pull gpt-oss:120b   etc. (docs/SPARK_SETUP.md)
# gemma4:31b (19GB on disk) leads gpt-oss:120b (65GB) here, not the other way
# round (2026-07-20): gpt-oss:120b alongside a warm VLM_MODELS pick (71GB for
# qwen3.6:35b-a3b-bf16) overcommits this Spark's 121GB unified memory — see
# the Gotchas page — so SparkLLM.warm() only pre-warms VLM_MODELS and leaves
# REPORT_MODELS to cold-load lazily on first /report request. Putting
# gemma4:31b first means that lazy load is small enough to be fast AND still
# small enough to warm safely alongside the VLM if warm() is ever extended to
# cover both again. gpt-oss:120b stays in the chain as a fallback, and as an
# explicit opt-in via REPORT_MODELS=gpt-oss:120b,... for anyone who wants its
# extra quality and is fine trading report latency for it.
REPORT_MODELS = [
    m.strip() for m in os.environ.get(
        "REPORT_MODELS", "gemma4:31b,qwen3:32b,gpt-oss:120b,llama3.3:70b,llava:7b"
    ).split(",") if m.strip()
]
VLM_MODELS = [
    m.strip() for m in os.environ.get(
        "VLM_MODELS",
        "gemma4:26b,qwen3.6:35b-a3b-bf16,qwen2.5vl:32b,gemma3:27b,llava:7b"
    ).split(",") if m.strip()
]

# --- Detection ------------------------------------------------------------
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "plan_a_yolov8" / "weights" / "best.pt"
LIVE_CONF = float(os.environ.get("LIVE_CONF", "0.4"))
LIVE_IMGSZ = int(os.environ.get("LIVE_IMGSZ", "640"))

# --- Hazard thresholds ------------------------------------------------------
# Proximity: fraction of the frame diagonal (see hazard_logic.find_proximity_hazards)
PROXIMITY_DISTANCE_RATIO = float(os.environ.get("PROXIMITY_DISTANCE_RATIO", "0.25"))
# Vehicle movement: centroid displacement per second as a fraction of frame diagonal
VEHICLE_MOVE_RATIO_PER_SEC = float(os.environ.get("VEHICLE_MOVE_RATIO_PER_SEC", "0.04"))
# Height (experimental heuristic): worker box bottom above this fraction of
# frame height counts as elevated. Disabled unless HEIGHT_ZONE_ENABLED=1.
HEIGHT_ZONE_ENABLED = os.environ.get("HEIGHT_ZONE_ENABLED", "0") == "1"
HEIGHT_ZONE_FRACTION = float(os.environ.get("HEIGHT_ZONE_FRACTION", "0.45"))

# --- PPE compliance (PPE_Detect/best.pt: Hardhat/Safety Vest/boots/gloves) -
# Baseline check on every detected worker, plus the cross-check that lets a
# height alarm suppress itself (see hazard_logic.find_ppe_violations and
# live.py IncidentAnalyst._verify_height). Has no harness class — that's
# still the VLM's job.
PPE_MODEL_PATH = PROJECT_ROOT / "PPE_Detect" / "best.pt"
PPE_ENABLED = os.environ.get("PPE_ENABLED", "1") == "1"
PPE_CONF = float(os.environ.get("PPE_CONF", "0.5"))
# Full-frame PPE_Detect pass runs on this interval, not every frame — it's a
# second YOLO model, so this caps the added inference cost.
PPE_CHECK_INTERVAL_SEC = float(os.environ.get("PPE_CHECK_INTERVAL_SEC", "2.0"))

# --- Alerts ----------------------------------------------------------------
# Consecutive detection passes a hazard must persist before the alarm fires
# (debounce against single-frame flickers), and the cooldown between repeats
# of the same alert so the voice does not spam the site.
ALERT_TRIGGER_FRAMES = int(os.environ.get("ALERT_TRIGGER_FRAMES", "3"))
# The FIRST alert of a hazard type is always instant regardless of this
# value — cooldown only spaces out REPEATS of an ongoing hazard. Bumped
# 6->8s (2026-07-19) to feel less spammy on busy scenes without dulling the
# on-the-spot reaction to a new hazard.
ALERT_COOLDOWN_SEC = float(os.environ.get("ALERT_COOLDOWN_SEC", "8"))
# On-the-spot AI analysis: min seconds between automatic live-report rewrites
REPORT_REFRESH_SEC = float(os.environ.get("REPORT_REFRESH_SEC", "45"))
# Report output length cap (tokens). Directly trades length for wall-clock
# time: on the Spark, gemma4:31b (REPORT_MODELS' default first choice)
# sustains ~9.3 tok/s once warm — measured 2026-07-20, see the Gotchas page —
# so 260 targets a condensed-but-complete report under ~30s; the previous
# 900 (a fuller, more elaborated report) took ~90-100s for the same content
# depth per section.
REPORT_NUM_PREDICT = int(os.environ.get("REPORT_NUM_PREDICT", "260"))
AUDIO_DIR = PROJECT_ROOT / "assets" / "audio"
EVENTS_LOG = PROJECT_ROOT / "logs" / "events.jsonl"

# --- Server-side voice (Piper neural TTS) -----------------------------------
# Browser Web Speech API quality is whatever's installed on each viewer's own
# device — on a bare Linux box that's espeak-ng (robotic), with nothing
# better to pick from no matter how the browser-side voice is chosen. Piper
# synthesizes server-side instead (~200ms for a short alert, one-time model
# load at startup) so every viewer gets the same natural voice regardless of
# their own device. Falls back to the browser's own voice automatically if
# disabled or the model file is missing — never blocks the dashboard.
TTS_ENABLED = os.environ.get("TTS_ENABLED", "1") == "1"
TTS_VOICE_PATH = PROJECT_ROOT / "assets" / "tts_voices" / os.environ.get(
    "TTS_VOICE", "en_US-ryan-medium.onnx")

# --- Live dashboard ----------------------------------------------------------
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8090"))
MJPEG_QUALITY = int(os.environ.get("MJPEG_QUALITY", "80"))
