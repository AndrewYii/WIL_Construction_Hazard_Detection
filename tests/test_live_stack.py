"""
Unit tests for the real-time stack: alert debounce/cooldown, hazard layer
(vehicle motion, height zone), Spark LLM client chain resolution, and the
report generator's template fallback. No GPU, camera, or Ollama needed.

Run:
    python -m pytest tests/test_live_stack.py -v
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "app"))

import config
from alerts import AlertEngine
from hazard_logic import Detection, VehicleMotionTracker, find_height_hazards
from llm_client import SparkLLM
from report_generation import generate_hazard_report


# --------------------------------------------------------------------------
# AlertEngine
# --------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make_engine(tmp_path, clock, trigger=3, cooldown=6.0):
    return AlertEngine(trigger_frames=trigger, cooldown_sec=cooldown,
                       events_path=tmp_path / "events.jsonl",
                       player=None, clock=clock)


def test_alert_debounce_requires_consecutive_frames(tmp_path):
    clock = FakeClock()
    engine = make_engine(tmp_path, clock)
    assert engine.update({"proximity"}) == []
    assert engine.update({"proximity"}) == []
    fired = engine.update({"proximity"})  # 3rd consecutive pass -> fires
    assert len(fired) == 1
    assert fired[0]["type"] == "proximity"


def test_alert_flicker_resets_debounce(tmp_path):
    clock = FakeClock()
    engine = make_engine(tmp_path, clock)
    engine.update({"proximity"})
    engine.update({"proximity"})
    engine.update(set())  # gap resets the streak
    engine.update({"proximity"})
    assert engine.update({"proximity"}) == []


def test_alert_cooldown_blocks_repeats(tmp_path):
    clock = FakeClock()
    engine = make_engine(tmp_path, clock, cooldown=6.0)
    for _ in range(3):
        engine.update({"proximity"})
    # hazard persists: within cooldown nothing more fires
    clock.t += 3
    assert engine.update({"proximity"}) == []
    # after cooldown it fires again
    clock.t += 4
    assert len(engine.update({"proximity"})) == 1
    assert engine.counts()["proximity"] == 2


def test_alert_events_logged_jsonl(tmp_path):
    import json
    clock = FakeClock()
    engine = make_engine(tmp_path, clock)
    for _ in range(3):
        engine.update({"vehicle"}, detail={"frame": 42})
    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "vehicle"
    assert event["frame"] == 42


def test_independent_hazard_types(tmp_path):
    clock = FakeClock()
    engine = make_engine(tmp_path, clock, trigger=1)
    fired = engine.update({"proximity", "height"})
    assert {e["type"] for e in fired} == {"proximity", "height"}


# --------------------------------------------------------------------------
# Hazard layer
# --------------------------------------------------------------------------

def _vehicle(x, y, w=100, h=80):
    return Detection(1, 0.9, (x - w / 2, y - h / 2, x + w / 2, y + h / 2))


def test_motion_tracker_flags_moving_vehicle():
    tracker = VehicleMotionTracker(frame_diag=1000, move_ratio_per_sec=0.04)
    tracker.update([_vehicle(100, 100)], now=0.0)
    moving = tracker.update([_vehicle(160, 100)], now=1.0)  # 60 px/s > 40 px/s
    assert len(moving) == 1


def test_motion_tracker_ignores_stationary_vehicle():
    tracker = VehicleMotionTracker(frame_diag=1000, move_ratio_per_sec=0.04)
    tracker.update([_vehicle(100, 100)], now=0.0)
    moving = tracker.update([_vehicle(105, 100)], now=1.0)  # 5 px/s, jitter
    assert moving == []


def test_motion_tracker_ignores_teleport_as_new_vehicle():
    tracker = VehicleMotionTracker(frame_diag=1000, move_ratio_per_sec=0.04)
    tracker.update([_vehicle(100, 100)], now=0.0)
    # 400px jump in one pass = association error / new vehicle, not motion
    moving = tracker.update([_vehicle(500, 100)], now=1.0)
    assert moving == []


def test_height_zone_flags_elevated_worker():
    high = Detection(0, 0.9, (100, 50, 150, 200))    # bottom y=200
    low = Detection(0, 0.9, (300, 500, 350, 700))    # bottom y=700
    vehicle = Detection(1, 0.9, (0, 0, 50, 100))     # vehicles never flagged
    flagged = find_height_hazards([high, low, vehicle], frame_height=1000,
                                  zone_fraction=0.45)
    assert flagged == [high]


# --------------------------------------------------------------------------
# Spark LLM client
# --------------------------------------------------------------------------

def test_resolve_picks_first_available(monkeypatch):
    client = SparkLLM(host="http://test:11434")
    monkeypatch.setattr(client, "available_models",
                        lambda cache_sec=0: ["llava:7b", "qwen3:32b"])
    assert client.resolve(["gpt-oss:120b", "qwen3:32b", "llava:7b"]) == "qwen3:32b"


def test_resolve_matches_bare_name_to_tagged(monkeypatch):
    client = SparkLLM(host="http://test:11434")
    monkeypatch.setattr(client, "available_models",
                        lambda cache_sec=0: ["gpt-oss:120b-cloud"])
    assert client.resolve(["gpt-oss:120b"]) == "gpt-oss:120b-cloud"


def test_resolve_none_when_server_empty(monkeypatch):
    client = SparkLLM(host="http://test:11434")
    monkeypatch.setattr(client, "available_models", lambda cache_sec=0: [])
    assert client.resolve(config.REPORT_MODELS) is None
    assert not client.is_up()


def test_resolve_vision_skips_text_only_models(monkeypatch):
    # mirrors the real Spark: gemma4 pulled but its build reports no vision
    client = SparkLLM(host="http://test:11434")
    monkeypatch.setattr(client, "available_models",
                        lambda cache_sec=0: ["gemma4:26b", "qwen2.5vl:32b"])
    client._caps = {"gemma4:26b": {"completion", "tools", "thinking"},
                    "qwen2.5vl:32b": {"completion", "vision"}}
    chain = ["gemma4:26b", "qwen2.5vl:32b"]
    assert client.resolve(chain) == "gemma4:26b"                # text use: fine
    assert client.resolve(chain, need="vision") == "qwen2.5vl:32b"


def test_resolve_unknown_capabilities_not_filtered(monkeypatch):
    # older Ollama servers omit capabilities — never filter on missing info
    client = SparkLLM(host="http://test:11434")
    monkeypatch.setattr(client, "available_models",
                        lambda cache_sec=0: ["gemma4:26b"])
    assert client.resolve(["gemma4:26b"], need="vision") == "gemma4:26b"


# --------------------------------------------------------------------------
# Report generation fallback
# --------------------------------------------------------------------------

SAMPLE_REPORT = {
    "video": "unit_test.mp4",
    "frames_processed": 100,
    "duration_sec": 4.0,
    "avg_fps": 25.0,
    "detections": {"worker": 5, "dangerous_vehicle": 2},
    "proximity_hazard_events": [{"frame": 10, "timestamp_sec": 0.4, "pairs": 1}],
    "total_hazard_events": 1,
}


def test_report_falls_back_to_template_when_server_down(monkeypatch):
    import llm_client
    monkeypatch.setattr(llm_client.SparkLLM, "generate_report", lambda self, p: None)
    text = generate_hazard_report(SAMPLE_REPORT)
    assert "## Site Safety Report" in text
    assert "PDPA" in text


def test_prompt_includes_vlm_incident_notes():
    from report_generation import _build_prompt
    report = dict(SAMPLE_REPORT)
    report["incident_notes"] = [
        {"iso": "2026-07-14 15:00:01", "type": "proximity",
         "note": "A worker in a green vest stands beside a reversing excavator."},
    ]
    prompt = _build_prompt(report)
    assert "EYEWITNESS SCENE NOTES" in prompt
    assert "reversing excavator" in prompt
    # without notes the section is absent
    assert "EYEWITNESS SCENE NOTES" not in _build_prompt(SAMPLE_REPORT)


def test_report_uses_llm_text_when_valid(monkeypatch):
    import llm_client
    llm_text = "<think>reasoning...</think>## Site Safety Report\n\nAll clear."
    monkeypatch.setattr(llm_client.SparkLLM, "generate_report", lambda self, p: llm_text)
    text = generate_hazard_report(SAMPLE_REPORT)
    assert text.startswith("## Site Safety Report")
    assert "<think>" not in text
