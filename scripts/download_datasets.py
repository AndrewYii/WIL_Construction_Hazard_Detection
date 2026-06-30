"""
Download remaining training datasets:
  - ConstructionSite10k  (HuggingFace: LouisChen15/ConstructionSite)
  - Roboflow v27         (roboflow-universe-projects/construction-site-safety)
  - Construction-Hazard-Detection (object-detection-qn97p/construction-hazard-detection)

Usage:
    set ROBOFLOW_API_KEY=your_key_here
    python scripts/download_datasets.py --dst data/raw
"""

import argparse
import os
import shutil
from pathlib import Path


def download_constructionsite10k(dst: Path):
    print("\n[ConstructionSite10k] Downloading from HuggingFace ...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub not installed. Run: pip install huggingface_hub")

    out = dst / "constructionsite10k"
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="LouisChen15/ConstructionSite",
        repo_type="dataset",
        local_dir=str(out),
    )
    print(f"  Saved to {out}")


def download_roboflow(api_key: str, workspace: str, project: str, version: int, dst: Path, fmt: str = "yolov8"):
    print(f"\n[Roboflow] {workspace}/{project} v{version} ...")
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("roboflow not installed. Run: pip install roboflow")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(fmt, location=str(dst), overwrite=True)
    print(f"  Saved to {dst}")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", default="data/raw", help="Root folder for raw datasets")
    parser.add_argument("--skip-hf", action="store_true", help="Skip ConstructionSite10k download")
    parser.add_argument("--skip-roboflow", action="store_true", help="Skip Roboflow downloads")
    args = parser.parse_args()

    dst = Path(args.dst)

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key and not args.skip_roboflow:
        raise SystemExit(
            "ROBOFLOW_API_KEY not set.\n"
            "Run:  set ROBOFLOW_API_KEY=your_key_here\n"
            "Then re-run this script."
        )

    if not args.skip_hf:
        download_constructionsite10k(dst)

    if not args.skip_roboflow:
        download_roboflow(
            api_key=api_key,
            workspace="roboflow-universe-projects",
            project="construction-site-safety",
            version=27,
            dst=dst / "roboflow_v27",
        )
        download_roboflow(
            api_key=api_key,
            workspace="object-detection-qn97p",
            project="construction-hazard-detection",
            version=92,
            dst=dst / "hazard_detection",
        )

    print("\nAll downloads complete.")
    print("Next: run the convert_*.py scripts, then merge_and_split.py")


if __name__ == "__main__":
    main()
