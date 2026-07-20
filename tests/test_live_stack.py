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
from hazard_logic import (Detection, VehicleMotionTracker, find_height_hazards,
                          find_ppe_violations)
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


def test_dynamic_message_overrides_default(tmp_path):
    clock = FakeClock()
    engine = make_engine(tmp_path, clock, trigger=1)
    fired = engine.update({"proximity"},
                          messages={"proximity": "Warning! Two workers too "
                                                 "close to vehicle on the left."})
    assert "on the left" in fired[0]["message"]
    # missing override falls back to the fixed phrase
    clock.t += 100
    fired = engine.update({"proximity"}, messages={})
    assert fired[0]["message"].startswith("Warning! Worker too close")


def test_fire_now_respects_cooldown(tmp_path):
    clock = FakeClock()
    engine = make_engine(tmp_path, clock, cooldown=6.0)
    assert engine.fire_now("height", "Worker at height without PPE.") is not None
    assert engine.fire_now("height") is None          # cooling down
    clock.t += 7
    assert engine.fire_now("height") is not None
    assert engine.counts()["height"] == 2


def test_parse_ppe_verdicts():
    from live import parse_ppe
    assert parse_ppe('{"hardhat": true, "harness": false}') == {
        "hardhat": True, "harness": False}
    assert parse_ppe('Sure! {"hardhat": false, "harness": false} done') == {
        "hardhat": False, "harness": False}
    assert parse_ppe("I cannot tell from this image.") is None
    assert parse_ppe(None) is None


def test_list_local_cameras_plain_single_webcam():
    """A single, ordinary webcam (no duplicate-name siblings) must appear
    exactly as reported, with no format suffix or reordering noise added —
    the multi-node handling must be a no-op for the common case."""
    import platform
    from unittest.mock import mock_open, patch

    import live

    if platform.system() == "Windows":
        return  # this test exercises the Linux sysfs enumeration path

    live._CAM_CACHE.update(t=0.0, list=[])
    names = {"/sys/class/video4linux/video0/name": "HP TrueVision HD\n"}
    with patch("glob.glob", return_value=list(names)), \
         patch("builtins.open", mock_open(read_data=""), create=True) as m_open, \
         patch.object(live, "_v4l2_probe", return_value=(True, "YUYV 4:2:2")):
        m_open.side_effect = lambda path, *a, **k: mock_open(
            read_data=names[path])()
        cams = live.list_local_cameras()
    assert cams == [{"index": 0, "label": "HP TrueVision HD", "usable": True}]


def test_list_local_cameras_multi_node_device_alongside_plain_webcam():
    """A multi-node camera (RealSense-shaped: several capture nodes sharing
    one name, some not actually capturable) coexisting with an ordinary
    single-node webcam — proves the filtering/labeling is generic, not
    RealSense-specific string matching. Regression test for the bug found
    2026-07-19 (all cameras silently reported as index 4)."""
    import platform
    from unittest.mock import mock_open, patch

    import live

    if platform.system() == "Windows":
        return

    live._CAM_CACHE.update(t=0.0, list=[])
    names = {
        "/sys/class/video4linux/video0/name": "HP TrueVision HD\n",
        "/sys/class/video4linux/video1/name": "Depth Module\n",
        "/sys/class/video4linux/video2/name": "Depth Module\n",
        "/sys/class/video4linux/video3/name": "Depth Module\n",
        "/sys/class/video4linux/video4/name": "Depth Module\n",
    }
    # video1: depth (capturable, mono) · video2: metadata-only (not
    # capturable) · video3: infrared (capturable, mono) · video4: color
    # (capturable) — same shape as the real RealSense probe results.
    probe_results = {
        0: (True, "YUYV 4:2:2"),
        1: (True, "16-bit Depth"),
        2: (False, ""),
        3: (True, "8-bit Greyscale"),
        4: (True, "YUYV 4:2:2"),
    }
    with patch("glob.glob", return_value=sorted(names)), \
         patch("builtins.open", create=True) as m_open, \
         patch.object(live, "_v4l2_probe", side_effect=lambda i: probe_results[i]):
        m_open.side_effect = lambda path, *a, **k: mock_open(
            read_data=names[path])()
        cams = live.list_local_cameras()

    by_index = {c["index"]: c for c in cams}
    assert by_index[0]["label"] == "HP TrueVision HD"  # untouched, no duplicates
    assert by_index[0]["usable"] is True
    assert 2 not in by_index  # metadata-only node dropped
    assert "recommended" in by_index[4]["label"]
    assert by_index[4]["usable"] is True
    assert "not usable here, no color" in by_index[1]["label"]
    assert by_index[1]["usable"] is False  # the dashboard must refuse to let this be clicked
    assert "not usable here, no color" in by_index[3]["label"]
    assert by_index[3]["usable"] is False
    # the color node sorts before its non-color siblings
    indices = [c["index"] for c in cams]
    assert indices.index(4) < indices.index(1)
    assert indices.index(4) < indices.index(3)


def test_public_source_hides_internal_paths():
    from live import public_source
    assert public_source(0) == "0"
    assert public_source("C:/Users/User/Desktop/secret/site_demo.mp4") == "site_demo.mp4"
    assert public_source("/home/spark/clips/yard.mp4") == "yard.mp4"
    assert public_source("http://192.168.0.5:8080/video?token=abc") == "http://192.168.0.5"
    assert public_source(None) == ""


def test_disarm_clears_current_so_camera_on_cannot_resume_stale_source():
    """Regression test (2026-07-20): entering Camera mode with no usable
    camera found used to leave `current` pointed at whatever was active
    before (e.g. an uploaded video), so a later camera-on press silently
    replayed that video. disarm() must clear it so camera-on is a no-op."""
    from live import LatestFrame, SourceManager
    sources = SourceManager(LatestFrame())
    sources.arm("/tmp/siteguard_upload/site_demo.mp4", loop_file=True)
    sources.disarm()
    assert sources.current is None
    assert sources.camera_on is False
    sources.set_camera(True)
    assert sources.current is None
    assert sources.camera_on is False


def test_arm_stops_a_previously_running_capture_thread():
    """Regression test (2026-07-20): entering Camera mode while an uploaded
    video's CaptureThread was still running left it running in the
    background — `current`/`camera_on` updated to look like standby, but
    the video kept looping, feeding frames, and firing genuinely new
    alerts into the freshly-cleared log the whole time, invisible only
    because the client hides the still-updating view behind a placeholder.
    arm() must stop whatever capture is active, exactly like start() and
    use_browser() already do — it only ever meant to skip OPENING a new
    device, not skip stopping the old one."""
    import threading

    from live import LatestFrame, SourceManager

    sources = SourceManager(LatestFrame())

    class FakeCapture:
        def __init__(self):
            self.stop_flag = threading.Event()
            self.close_on_exit = True

        def is_alive(self):
            return not self.stop_flag.is_set()

    fake = FakeCapture()
    sources.capture = fake
    sources.camera_on = True
    sources.current = "/tmp/siteguard_upload/site_demo.mp4"

    sources.arm(4)  # switching to the Live Camera tab, arming the RealSense

    assert fake.stop_flag.is_set()  # the video's capture must be told to stop
    assert sources.capture is None
    assert sources.current == 4
    assert sources.camera_on is False


def test_claiming_browser_camera_clears_old_alerts_like_every_other_source_change():
    """Regression test (2026-07-20): every source-change entry point
    (/upload, /switch, /arm) calls state.reset_session() so the dashboard's
    alert log/voice/active-hazard state starts clean on the new source.
    /ingest?claim=1 (the "Use this device camera" button) was the one left
    out — the old source's fired alerts and active hazards leaked into the
    new stream, which is what the user was hearing/seeing. This asserts the
    reset primitive the fix relies on actually clears what the dashboard
    reads back via SessionState.snapshot()."""
    from live import SessionState
    from alerts import AlertEngine

    state = SessionState(AlertEngine())
    state.engine.fire_now("proximity", "Warning! Worker too close to vehicle.")
    with state.lock:
        state.active_hazards = ["proximity"]
        state.workers_now = 2

    snap_before = state.snapshot()
    assert snap_before["recent_alerts"]
    assert snap_before["active_hazards"] == ["proximity"]

    state.reset_session()  # what the fixed /ingest claim branch now calls

    snap_after = state.snapshot()
    assert snap_after["recent_alerts"] == []
    assert snap_after["active_hazards"] == []
    assert snap_after["workers"] == 0


def test_detection_loop_survives_a_transient_predictor_crash(monkeypatch):
    """Regression test (2026-07-20): a single detector.predict_frame()
    exception (the real trigger was a CUDA OOM from a warm Ollama model
    still holding unified memory on the Spark) used to be unhandled and
    killed the whole detection thread — nothing restarts a dead thread, so
    the dashboard was stuck on the standby placeholder forever afterward,
    indistinguishable from an actual unarmed-camera problem. The loop must
    catch it, record state.detector_error, and keep processing later frames
    on the same thread."""
    import threading
    import time as time_mod

    import numpy as np
    import live
    from alerts import AlertEngine
    from live import LatestFrame, SessionState, detection_loop

    monkeypatch.setattr(live.time, "sleep", lambda *_: None)  # skip the real backoff

    class FlakyDetector:
        def __init__(self):
            self.calls = 0

        def predict_frame(self, frame, conf, imgsz):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("CUDA error: out of memory")
            return []

    class Args:
        conf = 0.4
        imgsz = 640
        height_zone = False

    in_slot, out_slot = LatestFrame(), LatestFrame()
    state = SessionState(AlertEngine())
    stop_flag = threading.Event()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    t = threading.Thread(target=detection_loop, args=(
        Args(), FlakyDetector(), in_slot, out_slot, state, state.engine, stop_flag))
    t.start()
    try:
        in_slot.put(frame)  # first call raises — must not kill the thread
        deadline = time_mod.time() + 2
        while state.detector_error is None and time_mod.time() < deadline:
            time_mod.sleep(0.01)
        assert state.detector_error is not None
        assert "out of memory" in state.detector_error
        assert t.is_alive()

        in_slot.put(frame)  # second call succeeds on the same thread
        deadline = time_mod.time() + 2
        while state.frames == 0 and time_mod.time() < deadline:
            time_mod.sleep(0.01)
        assert state.frames >= 1
        assert state.detector_error is None
    finally:
        stop_flag.set()
        in_slot.close()
        t.join(timeout=2)


def test_annotate_marks_ppe_violator_red_despite_stale_box():
    """ppe_violations is held over from the last throttled PPE_Detect pass
    (up to PPE_CHECK_INTERVAL_SEC old) so its Detection is never == this
    frame's freshly re-detected one (conf/xyxy jitter every frame) — the
    violator box must still be matched and drawn red by spatial overlap,
    not identity. Regression test for the bug found 2026-07-19."""
    import numpy as np
    from live import BOX_COLORS, PPE_VIOLATION_COLOR, annotate

    frame = np.full((400, 600, 3), 220, dtype=np.uint8)
    current_worker = Detection(0, 0.87, (100, 50, 200, 300))
    compliant_worker = Detection(0, 0.91, (350, 60, 450, 310))
    stale_violation = Detection(0, 0.79, (98, 48, 198, 298))  # same person, earlier pass

    annotated = annotate(frame.copy(), [current_worker, compliant_worker], ["ppe"],
                         ppe_violations=[stale_violation])

    assert annotated[50, 150].tolist() == list(PPE_VIOLATION_COLOR)
    assert annotated[60, 400].tolist() == list(BOX_COLORS[0])


def test_verify_height_discards_result_from_a_stale_session():
    """A VLM round trip for height verification can take several seconds.
    If a tab switch / new upload / new camera started a fresh session while
    it was in flight, the result belongs to a session that no longer exists
    and must be discarded — not fired as a "new" alert on top of an
    already-reset dashboard. Regression test for the bug found 2026-07-19
    (a stale height alert appeared, with working voice, moments after
    switching away from the video it was actually about)."""
    import numpy as np

    from live import IncidentAnalyst, SessionState

    engine = AlertEngine(events_path=None)
    state = SessionState(engine)
    analyst = IncidentAnalyst(state, engine)
    crop = np.zeros((10, 10, 3), dtype="uint8")

    submitted_epoch = state.session_epoch
    state.reset_session()  # a new session starts while the "VLM call" is still in flight
    assert state.session_epoch != submitted_epoch

    # client=None would crash if _verify_height touched it — proves the
    # stale-epoch check short-circuits before doing any of that work
    analyst._verify_height(client=None, message="test", crop=crop, epoch=submitted_epoch)

    assert state.incidents == []
    assert engine.counts()["height"] == 0


def test_verify_height_fires_normally_when_epoch_still_current():
    """Same session throughout (the common case) — verification proceeds
    and an unconfirmed candidate still alarms, safety-first."""
    import numpy as np

    from live import IncidentAnalyst, SessionState

    class _FakeClient:
        def is_up(self):
            return False  # VLM unreachable -> verdict stays None -> alarm fires

    engine = AlertEngine(events_path=None)
    state = SessionState(engine)
    analyst = IncidentAnalyst(state, engine)
    crop = np.zeros((10, 10, 3), dtype="uint8")

    analyst._verify_height(client=_FakeClient(), message="Warning! test",
                           crop=crop, epoch=state.session_epoch)

    assert len(state.incidents) == 1
    assert engine.counts()["height"] == 1


def test_compose_alert_messages_describes_scene():
    from live import compose_alert_messages
    worker_a = Detection(0, 0.9, (50, 100, 90, 260))
    worker_b = Detection(0, 0.9, (120, 100, 160, 260))
    vehicle = Detection(1, 0.9, (10, 80, 200, 300))     # left third of frame
    msgs = compose_alert_messages(
        hazard_pairs=[(worker_a, vehicle), (worker_b, vehicle)],
        moving_vehicles=[vehicle], elevated=[],
        workers=[worker_a, worker_b], frame_w=1280)
    assert msgs["proximity"] == ("Warning! 2 workers too close to vehicle "
                                 "on the left. Move away now.")
    assert "Vehicle moving on the left, 2 workers nearby." in msgs["vehicle"]
    assert "height" not in msgs


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


def test_height_relative_rule_needs_two_workers():
    # one worker high in the frame, zone rule off: nothing to compare against
    solo = Detection(0, 0.9, (100, 50, 150, 200))
    assert find_height_hazards([solo], 1000, use_zone=False) == []
    # add a ground-level worker: the elevated one is flagged relative to them
    ground = Detection(0, 0.9, (400, 600, 460, 900))
    flagged = find_height_hazards([solo, ground], 1000, use_zone=False)
    assert flagged == [solo]
    # two workers at similar level: no flag (gap under 30% of frame height)
    near = Detection(0, 0.9, (500, 550, 560, 820))
    assert find_height_hazards([ground, near], 1000, use_zone=False) == []


def test_ppe_violation_flags_worker_overlapping_no_hardhat():
    worker = Detection(0, 0.9, (100, 50, 150, 200))
    ppe_dets = [{"name": "NO-Hardhat", "conf": 0.8, "xyxy": (105, 50, 145, 90)}]
    assert find_ppe_violations([worker], ppe_dets) == [worker]


def test_ppe_no_violation_when_hardhat_confirmed():
    worker = Detection(0, 0.9, (100, 50, 150, 200))
    ppe_dets = [{"name": "Hardhat", "conf": 0.9, "xyxy": (105, 50, 145, 90)}]
    assert find_ppe_violations([worker], ppe_dets) == []


def test_ppe_missed_detection_is_not_a_violation():
    # PPE model saw nothing on this worker at all — absence isn't an alarm,
    # only an explicit NO-Hardhat/NO-Safety Vest hit is
    worker = Detection(0, 0.9, (100, 50, 150, 200))
    assert find_ppe_violations([worker], []) == []


def test_ppe_violation_ignores_non_overlapping_detection():
    worker = Detection(0, 0.9, (100, 50, 150, 200))
    # a NO-Hardhat box far away belongs to a different, undetected worker
    ppe_dets = [{"name": "NO-Hardhat", "conf": 0.8, "xyxy": (900, 900, 950, 950)}]
    assert find_ppe_violations([worker], ppe_dets) == []


def test_ppe_correlates_vest_box_below_worker_box():
    # real geometry from test_videos/site_overview.mp4: PPE_Detect's vest
    # box sits entirely below the worker detector's box for the same person
    # (different models, different body-region conventions) — x-ranges
    # nearly coincide, y-ranges don't overlap at all
    worker = Detection(0, 0.9, (1442, 562, 1501, 704))
    ppe_dets = [{"name": "NO-Safety Vest", "conf": 0.6, "xyxy": (1462, 792, 1503, 857)}]
    assert find_ppe_violations([worker], ppe_dets) == [worker]


def test_ppe_does_not_correlate_across_different_x_position():
    # a NO-Safety Vest detection on a different worker entirely (no
    # horizontal overlap) must not attach to this one, even if "below" it
    worker = Detection(0, 0.9, (1344, 549, 1404, 700))
    ppe_dets = [{"name": "NO-Safety Vest", "conf": 0.6, "xyxy": (1671, 764, 1706, 817)}]
    assert find_ppe_violations([worker], ppe_dets) == []


# --------------------------------------------------------------------------
# Spark LLM client
# --------------------------------------------------------------------------

def test_resolve_picks_first_available(monkeypatch):
    client = SparkLLM(host="http://test:11434")
    monkeypatch.setattr(client, "available_models",
                        lambda cache_sec=0: ["llava:7b", "qwen3:32b"])
    assert client.resolve(["gpt-oss:120b", "qwen3:32b", "llava:7b"]) == "qwen3:32b"


def test_warm_loads_both_report_and_vlm_models(monkeypatch):
    """warm() pre-loads both chains so neither pays a cold-load on first
    real use. Only safe on the Spark because REPORT_MODELS' default first
    choice (gemma4:31b, ~19GB) is small enough to sit alongside the resolved
    VLM model (~71GB) well within 121GB total unified memory — see the
    REPORT_MODELS comment in config.py and the 2026-07-20 Gotchas entry for
    why this used to OOM-kill the whole process when REPORT_MODELS defaulted
    to gpt-oss:120b (~65GB) instead."""
    import config
    client = SparkLLM(host="http://test:11434")
    monkeypatch.setattr(client, "available_models",
                        lambda cache_sec=0: [config.REPORT_MODELS[0], config.VLM_MODELS[0]])
    monkeypatch.setattr(client, "_caps", {config.VLM_MODELS[0]: {"vision"}})
    calls = []

    class FakeClient:
        def generate(self, model, prompt, think, keep_alive):
            calls.append(model)

    monkeypatch.setattr(client, "_get_client", lambda: FakeClient())
    client.warm()
    assert calls == [config.REPORT_MODELS[0], config.VLM_MODELS[0]]


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
