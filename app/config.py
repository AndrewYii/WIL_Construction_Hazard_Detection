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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Ollama server (NVIDIA DGX Spark) ------------------------------------
# Set OLLAMA_HOST (or SPARK_OLLAMA_HOST) to the Spark's address, e.g.
#   OLLAMA_HOST=http://192.168.1.50:11434
# The placeholder below assumes the hostname "spark" resolves on your LAN.
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
REPORT_MODELS = [
    m.strip() for m in os.environ.get(
        "REPORT_MODELS", "gpt-oss:120b,gemma4:31b,qwen3:32b,llama3.3:70b,llava:7b"
    ).split(",") if m.strip()
]
VLM_MODELS = [
    m.strip() for m in os.environ.get(
        "VLM_MODELS", "gemma4:26b,qwen2.5vl:32b,gemma3:27b,llava:7b"
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

# --- Alerts ----------------------------------------------------------------
# Consecutive detection passes a hazard must persist before the alarm fires
# (debounce against single-frame flickers), and the cooldown between repeats
# of the same alert so the voice does not spam the site.
ALERT_TRIGGER_FRAMES = int(os.environ.get("ALERT_TRIGGER_FRAMES", "3"))
ALERT_COOLDOWN_SEC = float(os.environ.get("ALERT_COOLDOWN_SEC", "6"))
AUDIO_DIR = PROJECT_ROOT / "assets" / "audio"
EVENTS_LOG = PROJECT_ROOT / "logs" / "events.jsonl"

# --- Live dashboard ----------------------------------------------------------
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8090"))
MJPEG_QUALITY = int(os.environ.get("MJPEG_QUALITY", "80"))
