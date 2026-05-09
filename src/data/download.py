"""
download.py
-----------
Downloads the OpenI (Indiana University) chest X-ray dataset.

OpenI contains:
  - 7,470 chest X-ray images (PNG)
  - 3,955 radiology reports (XML)
  - Paired image + report data

Usage:
  python src/data/download.py
"""

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

# ------------------------------------------------------------------ #
#  PATHS  (edit RAW_DIR if you want data somewhere else)             #
# ------------------------------------------------------------------ #
RAW_DIR   = Path("data/raw/openI")
IMAGE_DIR = RAW_DIR / "images"
REPORT_DIR = RAW_DIR / "reports"

# ------------------------------------------------------------------ #
#  OPENI DOWNLOAD URLS                                               #
# ------------------------------------------------------------------ #
# Direct download links from the NIH OpenI collection
URLS = {
    "images":  "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz",
    "reports": "https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz",
}


def download_file(url: str, dest_path: Path) -> None:
    """
    Downloads a file from url → dest_path with a progress bar.
    Skips download if file already exists.
    """
    if dest_path.exists():
        print(f"[SKIP] Already downloaded: {dest_path.name}")
        return

    print(f"[DOWNLOAD] {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"[DONE] Saved to {dest_path}")


def extract_tgz(tgz_path: Path, extract_to: Path) -> None:
    """
    Extracts a .tgz archive into extract_to directory.
    Skips if already extracted.
    """
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"[SKIP] Already extracted: {extract_to}")
        return

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"[EXTRACT] {tgz_path.name} → {extract_to}")

    import tarfile
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_to)

    print(f"[DONE] Extracted to {extract_to}")


def verify_download() -> None:
    """
    Prints a summary of what was downloaded.
    Helps you confirm everything is in place before preprocessing.
    """
    images  = list(IMAGE_DIR.glob("*.png"))
    reports = list(REPORT_DIR.glob("**/*.xml"))

    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Images  found : {len(images):,}  (expected ~7,470)")
    print(f"Reports found : {len(reports):,}  (expected ~3,955)")
    print(f"Image dir     : {IMAGE_DIR.resolve()}")
    print(f"Report dir    : {REPORT_DIR.resolve()}")

    if len(images) == 0:
        print("\n[WARNING] No images found. Download may have failed.")
    if len(reports) == 0:
        print("\n[WARNING] No reports found. Download may have failed.")

    if len(images) > 0 and len(reports) > 0:
        print("\n[OK] Dataset ready for preprocessing.")
        print("     Next step: python src/data/preprocess.py")


def main():
    print("="*50)
    print("OpenI Chest X-ray Dataset Downloader")
    print("="*50)

    # -- Step 1: Download archives --
    images_tgz  = RAW_DIR / "NLMCXR_png.tgz"
    reports_tgz = RAW_DIR / "NLMCXR_reports.tgz"

    download_file(URLS["images"],  images_tgz)
    download_file(URLS["reports"], reports_tgz)

    # -- Step 2: Extract archives --
    extract_tgz(images_tgz,  IMAGE_DIR)
    extract_tgz(reports_tgz, REPORT_DIR)

    # -- Step 3: Verify --
    verify_download()


if __name__ == "__main__":
    main()