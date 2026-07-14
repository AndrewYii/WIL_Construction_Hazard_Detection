"""
Automated end-to-end test for the full pipeline: video in -> Plan A detection
-> proximity hazard logic -> Phase 2 safety report out.

Run:
    python -m pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "app"))

from inference import process_video
from report_generation import generate_hazard_report

WEIGHTS = ROOT / "runs" / "detect" / "plan_a_yolov8" / "weights" / "best.pt"

REQUIRED_REPORT_KEYS = {
    "video", "frames_processed", "duration_sec", "avg_fps",
    "detections", "proximity_hazard_events", "total_hazard_events",
}


def _find_sample_video() -> Path:
    candidates = list((ROOT / "data").rglob("*.mp4")) if (ROOT / "data").exists() else []
    if candidates:
        return candidates[0]
    downloads = Path.home() / "Downloads"
    candidates = sorted(downloads.glob("*.mp4"))
    if candidates:
        return candidates[0]
    pytest.skip("No sample .mp4 found under data/ or ~/Downloads to test with")


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory):
    if not WEIGHTS.exists():
        pytest.skip(f"Plan A weights not found at {WEIGHTS}")
    video_path = _find_sample_video()
    out_dir = tmp_path_factory.mktemp("pipeline_test")
    output_path = out_dir / "annotated.mp4"

    report = process_video(
        video_path=str(video_path),
        model_path=str(WEIGHTS),
        output_path=str(output_path),
        conf=0.4,
        imgsz=480,
        device="0",
        frame_skip=3,
    )
    return report, output_path


def test_report_has_expected_shape(pipeline_output):
    report, _ = pipeline_output
    assert REQUIRED_REPORT_KEYS.issubset(report.keys())
    assert report["frames_processed"] > 0
    assert report["avg_fps"] > 0
    assert set(report["detections"].keys()) == {"worker", "dangerous_vehicle"}
    assert report["detections"]["worker"] >= 0
    assert report["detections"]["dangerous_vehicle"] >= 0
    assert report["total_hazard_events"] == len(report["proximity_hazard_events"])


def test_annotated_video_written(pipeline_output):
    _, output_path = pipeline_output
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_hazard_events_reference_valid_frames(pipeline_output):
    report, _ = pipeline_output
    for event in report["proximity_hazard_events"]:
        assert 0 <= event["frame"] < report["frames_processed"]
        assert event["timestamp_sec"] >= 0
        assert event["pairs"] >= 1


def test_safety_report_generation(pipeline_output):
    report, _ = pipeline_output
    safety_report = generate_hazard_report(report)
    assert isinstance(safety_report, str)
    assert len(safety_report.strip()) > 0
    assert str(report["detections"]["worker"]) in safety_report or "worker" in safety_report.lower()
