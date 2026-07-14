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
import torch

from detectors import DETECTOR_REGISTRY, create_detector
from inference import process_video
from report_generation import generate_hazard_report, hazard_intervals, report_to_pdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "plan_a_yolov8" / "weights" / "best.pt"


@st.cache_resource(show_spinner="Loading detector...")
def get_detector(approach: str, weights_path: str, device: str):
    return create_detector(approach, weights_path=weights_path, device=device)

st.set_page_config(page_title="Construction Hazard Detection", page_icon="🚧", layout="wide")

# --- Visual theme: white base with cyan accents (CSS only, no behavior changes) ---
CYAN = "#0891B2"          # interactive accent (buttons, links, live elements)
CYAN_DARK = "#0E7490"     # hover state
CYAN_DEEP = "#155E75"     # steel cyan for headings / structural labels
CYAN_TINT = "#EFF7F9"     # tinted panel background
INK = "#122A33"           # near-black slate for primary text
BORDER = "#C9DDE3"        # hairline rules
BORDER_SOFT = "#DFEBEF"   # lighter grid rules
TEXT_MUTED = "#4C6B76"
HAZARD = "#B42318"        # safety red — hazard counts / warnings only
HAZARD_TINT = "#FCF1EF"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');

    /* Thin cyan datum line across the very top of the app */
    [data-testid="stHeader"] {{
        background: {CYAN_DEEP};
        height: 6px;
        min-height: 6px;
    }}
    [data-testid="stHeader"] * {{
        display: none;
    }}

    /* Typography */
    html, body, [data-testid="stAppViewContainer"],
    .stMarkdown, p, li, label {{
        font-family: "Barlow", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        color: {INK};
    }}
    h1, h2, h3 {{
        font-family: "Barlow Condensed", "Segoe UI", Arial, sans-serif !important;
        text-transform: uppercase;
    }}
    h1 {{
        font-weight: 700 !important;
        letter-spacing: 0.03em;
        color: {INK} !important;
        line-height: 1.05 !important;
        padding-top: 0.2rem !important;
    }}
    [data-testid="stHeadingWithActionElements"] h3, h3 {{
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.14em;
        color: {CYAN_DEEP} !important;
        border-bottom: 1px solid {BORDER};
        padding-bottom: 0.45rem !important;
    }}
    h3::before {{
        content: "";
        display: inline-block;
        width: 9px;
        height: 9px;
        background: {CYAN};
        margin-right: 0.55rem;
        vertical-align: 6%;
    }}
    [data-testid="stCaptionContainer"] p {{
        color: {TEXT_MUTED};
    }}

    /* Primary button: solid cyan action plate */
    button[kind="primary"], [data-testid="stBaseButton-primary"] {{
        background-color: {CYAN} !important;
        color: #FFFFFF !important;
        font-family: "Barlow Condensed", "Segoe UI", Arial, sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        border: 1px solid {CYAN_DARK} !important;
        border-radius: 2px !important;
    }}
    button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
        background-color: {CYAN_DARK} !important;
        border-color: {CYAN_DEEP} !important;
    }}

    /* Secondary / download buttons: outlined */
    [data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button {{
        border: 1px solid {CYAN} !important;
        color: {CYAN_DEEP} !important;
        border-radius: 2px !important;
        font-family: "Barlow Condensed", "Segoe UI", Arial, sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600 !important;
    }}
    [data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] button:hover {{
        background-color: {CYAN_TINT} !important;
        border-color: {CYAN_DARK} !important;
    }}

    /* Metric status plates: white, hairline frame, condensed tabular numerals */
    [data-testid="stMetric"] {{
        background-color: #FFFFFF;
        border: 1px solid {BORDER};
        border-top: 3px solid {CYAN};
        border-radius: 2px;
        padding: 0.7rem 0.9rem 0.6rem 0.9rem;
    }}
    [data-testid="stMetricLabel"] p {{
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.68rem !important;
        font-weight: 600;
        color: {TEXT_MUTED} !important;
    }}
    [data-testid="stMetricValue"] {{
        font-family: "Barlow Condensed", "Segoe UI", Arial, sans-serif !important;
        font-variant-numeric: tabular-nums;
        color: {CYAN_DEEP} !important;
        font-weight: 700 !important;
    }}
    /* Hazard metric (third plate in the metric row) carries the safety accent */
    [data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] {{
        border-top-color: {HAZARD};
        background-color: {HAZARD_TINT};
        border-color: #EBD5D1;
    }}
    [data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetricValue"] {{
        color: {HAZARD} !important;
    }}

    /* File uploader dropzone: intake bay */
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {CYAN_TINT};
        border: 1px dashed {CYAN};
        border-radius: 2px;
    }}

    /* Expander: framed report panel with tinted header strip */
    [data-testid="stExpander"] details {{
        border: 1px solid {BORDER};
        border-radius: 2px;
    }}
    [data-testid="stExpander"] summary {{
        background-color: {CYAN_TINT};
        border-bottom: 1px solid {BORDER_SOFT};
    }}
    [data-testid="stExpander"] summary p {{
        font-family: "Barlow Condensed", "Segoe UI", Arial, sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 600;
        color: {CYAN_DEEP};
    }}

    /* Alerts: engineered status panels instead of default rounded bubbles */
    [data-testid="stAlert"] {{
        border-radius: 2px;
    }}
    [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]),
    [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) > div {{
        background-color: #FFFFFF !important;
        color: {TEXT_MUTED} !important;
    }}
    [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {{
        border: 1px dashed {BORDER} !important;
        padding: 0.35rem 0.25rem;
    }}
    [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) p {{
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.78rem;
        font-weight: 500;
    }}
    [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]),
    [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) > div {{
        background-color: {CYAN_TINT} !important;
    }}
    [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {{
        border: 1px solid {BORDER} !important;
    }}
    [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) p {{
        color: {CYAN_DEEP} !important;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }}
    [data-testid="stAlert"]:has([data-testid="stAlertContentError"]),
    [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) > div {{
        background-color: {HAZARD_TINT} !important;
    }}
    [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {{
        border: 1px solid {HAZARD} !important;
    }}
    [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) p {{
        color: {HAZARD} !important;
        font-weight: 600;
    }}

    /* Dividers */
    hr {{
        border-color: {BORDER};
    }}

    /* Progress bar */
    [data-testid="stProgress"] div[role="progressbar"] > div {{
        background-color: {CYAN};
    }}

    /* Report text area: spec-sheet mono on white */
    [data-testid="stTextArea"] textarea {{
        font-family: Consolas, "Courier New", monospace;
        font-size: 0.8rem;
        color: {INK};
        background-color: #FFFFFF;
        border-radius: 2px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:0.9rem;
                border-bottom:1px solid {BORDER}; padding:0.35rem 0 0.55rem 0;
                margin-bottom:0.2rem;">
        <span style="background:{CYAN_DEEP}; color:#FFFFFF; font-weight:600;
                     font-family:'Barlow Condensed','Segoe UI',Arial,sans-serif;
                     padding:3px 12px; border-radius:2px; font-size:0.78rem;
                     letter-spacing:0.22em; text-transform:uppercase;">
            Site Safety Monitor
        </span>
        <span style="color:{TEXT_MUTED}; font-size:0.72rem; letter-spacing:0.18em;
                     text-transform:uppercase; font-weight:500;">
            Phase 1 Detector &nbsp;&middot;&nbsp; YOLOv8 &nbsp;&middot;&nbsp; Worker / Vehicle / Proximity
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.title("Construction Hazard Detection")
st.caption("Upload site footage to detect workers, dangerous vehicles, and proximity hazards, "
           "then generate a structured safety report.")

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    approach = st.selectbox("Detection approach", options=list(DETECTOR_REGISTRY.keys()))
    if approach.startswith("Plan D"):
        st.caption("Plan D runs about one frame per second. Use a short clip.")

    weights_path = str(DEFAULT_WEIGHTS)
    device = "0" if torch.cuda.is_available() else "cpu"
    conf = 0.4
    imgsz = 480
    frame_skip = 2

    if uploaded is not None:
        st.video(uploaded)

        if st.button("Run detection", type="primary", use_container_width=True):
            if not Path(weights_path).exists():
                st.error(f"Weights not found: {weights_path}")
                st.stop()

            tmp_dir = Path(tempfile.mkdtemp())
            input_path = tmp_dir / uploaded.name
            input_path.write_bytes(uploaded.getvalue())
            output_path = tmp_dir / f"annotated_{uploaded.name}"

            progress = st.progress(0.0, text="Starting...")

            def update(pct):
                progress.progress(min(pct, 1.0), text=f"Processing... {pct*100:.0f}%")

            detector = get_detector(approach, weights_path, device)

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
                    detector=detector,
                )
            progress.empty()

            with st.spinner("Generating safety report..."):
                safety_report = generate_hazard_report(report)

            st.session_state["report"] = report
            st.session_state["safety_report"] = safety_report
            st.session_state["output_path"] = str(output_path)
            st.session_state.setdefault("runs", {})[approach] = {
                "video": report["video"],
                "workers": report["detections"]["worker"],
                "vehicles": report["detections"]["dangerous_vehicle"],
                "hazard_events": report["total_hazard_events"],
                "avg_fps": report["avg_fps"],
            }

with right:
    st.subheader("Output")
    if "report" not in st.session_state:
        st.info("Upload a video and run detection to see results here.")
    else:
        report = st.session_state["report"]
        safety_report = st.session_state["safety_report"]
        output_path = st.session_state["output_path"]

        st.success(f"{report.get('approach', 'Plan A')} — {report['avg_fps']} fps average, "
                   f"{report['frames_processed']} frames processed")
        st.video(output_path)

        n_intervals, exposure_sec = hazard_intervals(report)
        peak = report.get("peak", {})
        m1, m2, m3 = st.columns(3)
        m1.metric("Peak workers on screen", peak.get("worker", "-"))
        m2.metric("Peak vehicles on screen", peak.get("dangerous_vehicle", "-"))
        m3.metric("Unsafe proximity intervals", n_intervals)
        if n_intervals:
            st.caption(f"Total unsafe exposure: {exposure_sec}s across {n_intervals} interval(s). "
                       "Full per-frame event log is in the JSON download.")

        with st.expander("Safety report (Phase 2)", expanded=True):
            st.markdown(safety_report)

        want_download = st.checkbox("Export this report?")
        if want_download:
            d1, d2, d3 = st.columns(3)
            d1.download_button(
                "Safety report (PDF)",
                data=report_to_pdf(safety_report),
                file_name="site_safety_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            d2.download_button(
                "Safety report (TXT)",
                data=safety_report,
                file_name="site_safety_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
            d3.download_button(
                "Detection data (JSON)",
                data=json.dumps(report, indent=2),
                file_name="hazard_report.json",
                mime="application/json",
                use_container_width=True,
            )

runs = st.session_state.get("runs", {})
if len(runs) > 1:
    st.subheader("Approach comparison")
    st.caption("Same session runs across approaches. Re-upload the same video and switch "
               "the detection approach in the sidebar to add rows.")
    st.dataframe(
        [{"approach": k, **v} for k, v in runs.items()],
        use_container_width=True,
        hide_index=True,
    )
