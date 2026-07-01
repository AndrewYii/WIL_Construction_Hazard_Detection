"""
Streamlit GUI: upload a video, run the Plan A YOLOv8 detector, get back
an annotated video + a structured hazard report.

Run:
    streamlit run app/app.py
"""

import json
import tempfile
from pathlib import Path

import streamlit as st

from inference import process_video

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "plan_a_yolov8" / "weights" / "best.pt"

st.set_page_config(page_title="Construction Hazard Detection", layout="wide")
st.title("Construction Hazard Detection")
st.caption("Upload site footage to detect workers, dangerous vehicles, and proximity hazards.")

with st.sidebar:
    st.header("Settings")
    weights_path = st.text_input("Model weights", value=str(DEFAULT_WEIGHTS))
    device = st.selectbox("Device", options=["0", "cpu"], index=0)
    conf = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
    imgsz = st.select_slider("Inference image size", options=[320, 480, 640], value=480)
    frame_skip = st.slider("Frame skip (higher = faster, less accurate)", 0, 4, 2)

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded is not None:
    if not Path(weights_path).exists():
        st.error(f"Weights not found: {weights_path}")
        st.stop()

    tmp_dir = Path(tempfile.mkdtemp())
    input_path = tmp_dir / uploaded.name
    input_path.write_bytes(uploaded.read())
    output_path = tmp_dir / f"annotated_{uploaded.name}"

    if st.button("Run detection", type="primary"):
        progress = st.progress(0.0, text="Starting...")

        def update(pct):
            progress.progress(min(pct, 1.0), text=f"Processing... {pct*100:.0f}%")

        with st.spinner("Running inference..."):
            report = process_video(
                video_path=str(input_path),
                model_path=weights_path,
                output_path=str(output_path),
                conf=conf,
                imgsz=imgsz,
                device=device,
                frame_skip=frame_skip,
                progress_cb=update,
            )
        progress.empty()
        st.success(f"Done — {report['avg_fps']} fps average, {report['frames_processed']} frames")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Annotated video")
            st.video(str(output_path))
        with col2:
            st.subheader("Hazard report")
            st.metric("Worker detections", report["detections"]["worker"])
            st.metric("Vehicle detections", report["detections"]["dangerous_vehicle"])
            st.metric("Proximity hazard events", report["total_hazard_events"])
            st.download_button(
                "Download report (JSON)",
                data=json.dumps(report, indent=2),
                file_name="hazard_report.json",
                mime="application/json",
            )
            with st.expander("Raw report"):
                st.json(report)
