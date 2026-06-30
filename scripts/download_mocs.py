"""
Download MOCS dataset files from Google Drive using gdown.

Downloads:
  - Train set images  (may have issues — script validates after download)
  - Train annotations (COCO JSON)
  - Test set images
  - Test annotation info (image_info_test)

Skips MOCS validation set entirely (val annotations will not be used).

Install dependency first:
    pip install gdown

Usage:
    python scripts/download_mocs.py --dst data/raw/mocs
"""

import argparse
import zipfile
import tarfile
from pathlib import Path

try:
    import gdown
except ImportError:
    raise SystemExit("gdown not installed. Run: pip install gdown")


FILES = {
    # Train set skipped — download link broken
    "val_images": {
        "id": "1Zeqr7C5p-hWNw5ta2fvD1bgLnXd-17fW",
        "filename": "val_images.zip",
        "note": "Validation set images (used as extra training data)",
    },
    "val_annotations": {
        "id": "18Q7ugoRJZ8ntjM07pwqkDhWGxPvQAgBt",
        "filename": "annotations_val.zip",
        "note": "Validation set COCO JSON annotations",
    },
    "test_images": {
        "id": "1Uj9-oZFIAk9Jy_JGMfNZlLbERplEXj-i",
        "filename": "test_images.zip",
        "note": "Test set images",
    },
    "test_image_info": {
        "id": "1yYJOdXF9SbuvU_BcVsuLb-mrgmUL5Pmj",
        "filename": "image_info_test.zip",
        "note": "Test set image metadata / annotations",
    },
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def download_file(file_id: str, dst_path: Path):
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading -> {dst_path.name} ...")
    gdown.download(url, str(dst_path), quiet=False)


def detect_format(path: Path) -> str:
    with open(path, "rb") as f:
        sig = f.read(8)
    if sig[:4] == b"PK\x03\x04":
        return "zip"
    if sig[:3] == b"Rar":
        return "rar"
    if sig[:2] == b"\x1f\x8b":
        return "gz"
    if sig[:5] == b"7z\xbc\xaf\x27":
        return "7z"
    return "unknown"


def extract(archive: Path, dst_dir: Path):
    fmt = detect_format(archive)
    print(f"  Extracting {archive.name} (detected: {fmt}) ...")
    if fmt == "zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dst_dir)
    elif fmt == "gz":
        with tarfile.open(archive) as t:
            t.extractall(dst_dir)
    elif fmt == "rar":
        seven_zip_paths = [
            r"C:\Program Files\7-Zip\7z.exe",
            r"C:\Program Files (x86)\7-Zip\7z.exe",
        ]
        seven_zip = next((p for p in seven_zip_paths if Path(p).exists()), None)
        if seven_zip:
            import subprocess
            result = subprocess.run(
                [seven_zip, "x", str(archive), f"-o{dst_dir}", "-y"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"  7-Zip error: {result.stderr}")
            else:
                print("  Extracted with 7-Zip.")
        else:
            print("  RAR file detected but 7-Zip not found.")
            print("  Install 7-Zip from https://www.7-zip.org/ then re-run with --skip-download")
            print(f"  Or extract manually: {archive} -> {dst_dir}")
    else:
        # May be a raw JSON file with wrong extension — try to parse it
        try:
            import json
            with open(archive, "r", encoding="utf-8") as f:
                json.load(f)
            out = dst_dir / archive.stem
            if not out.suffix:
                out = dst_dir / (archive.stem + ".json")
            import shutil
            shutil.copy2(archive, out)
            print(f"  Detected raw JSON — copied as {out.name}")
        except Exception:
            print(f"  Unknown format for {archive.name} — extract manually")


def validate_images(img_dir: Path) -> tuple[int, int]:
    """Return (ok_count, bad_count). Tries to open each image."""
    try:
        from PIL import Image
    except ImportError:
        print("  Pillow not installed — skipping image validation. pip install Pillow")
        return 0, 0

    ok = bad = 0
    bad_files = []
    for f in img_dir.rglob("*"):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        try:
            with Image.open(f) as im:
                im.verify()
            ok += 1
        except Exception:
            bad += 1
            bad_files.append(f)

    if bad_files:
        report = img_dir.parent / "bad_images.txt"
        report.write_text("\n".join(str(p) for p in bad_files))
        print(f"  Bad/corrupt images: {bad} — list saved to {report}")
    return ok, bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", default="data/raw/mocs", help="Destination folder")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, just extract existing zips")
    parser.add_argument("--no-validate", action="store_true", help="Skip image validation step")
    args = parser.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for key, info in FILES.items():
        archive_path = dst / info["filename"]

        if not args.skip_download:
            if archive_path.exists():
                print(f"  {info['filename']} already exists, skipping download")
            else:
                print(f"\n[{key}] {info['note']}")
                download_file(info["id"], archive_path)
        else:
            if not archive_path.exists():
                print(f"  {archive_path} not found, skipping")
                continue

        extract(archive_path, dst)

    if not args.no_validate:
        val_img_dir = dst / "val"
        if val_img_dir.exists():
            print(f"\nValidating val images in {val_img_dir} ...")
            ok, bad = validate_images(val_img_dir)
            if ok or bad:
                print(f"  OK: {ok} | Corrupt/bad: {bad}")

    print("\nDone. Run: python scripts/convert_mocs.py --src data/raw/mocs --dst data/raw/mocs/yolo_labels")
    print("       (converts val -> train pool, test -> processed/test)")


if __name__ == "__main__":
    main()
