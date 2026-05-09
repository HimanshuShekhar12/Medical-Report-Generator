"""
preprocess.py
-------------
Preprocesses raw OpenI chest X-ray images into model-ready format.

What this does to each image:
  1. Load PNG (grayscale chest X-ray)
  2. CLAHE  → enhance local contrast (standard in medical imaging)
  3. Resize → 256×256 pixels
  4. Normalize → pixel values to [0, 1]
  5. Save as PNG to data/processed/images/

CLAHE (Contrast Limited Adaptive Histogram Equalization):
  - Standard technique for medical X-ray enhancement
  - Makes subtle findings (nodules, infiltrates) more visible
  - Does NOT alter diagnostic content, only contrast

Usage:
  python src/data/preprocess.py
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import csv

# ------------------------------------------------------------------ #
#  PATHS                                                             #
# ------------------------------------------------------------------ #
RAW_IMAGE_DIR  = Path("data/raw/openI/images")
OUT_IMAGE_DIR  = Path("data/processed/images")
LABELS_CSV     = Path("data/processed/labels.csv")

# ------------------------------------------------------------------ #
#  PREPROCESSING CONFIG                                              #
# ------------------------------------------------------------------ #
IMAGE_SIZE    = 256          # resize to 256×256
CLAHE_CLIP    = 2.0          # CLAHE clip limit (higher = more contrast)
CLAHE_GRID    = (8, 8)       # CLAHE tile grid size


# ------------------------------------------------------------------ #
#  SINGLE IMAGE PREPROCESSING                                        #
# ------------------------------------------------------------------ #
def preprocess_image(image_path: Path, out_path: Path) -> bool:
    """
    Preprocesses one chest X-ray image.

    Steps:
      1. Load as grayscale (X-rays are single-channel)
      2. Apply CLAHE (contrast enhancement)
      3. Resize to IMAGE_SIZE × IMAGE_SIZE
      4. Normalize to [0, 255] uint8 (saves storage vs float32)
      5. Save to out_path

    Returns True if successful, False if image is corrupt/missing.
    """
    # -- Load image --
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    # -- CLAHE contrast enhancement --
    # Creates a CLAHE object with clip limit and tile size
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=CLAHE_GRID
    )
    img = clahe.apply(img)

    # -- Resize to 256×256 --
    # INTER_LANCZOS4 = best quality for downsampling medical images
    img = cv2.resize(
        img,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LANCZOS4
    )

    # -- Save --
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


# ------------------------------------------------------------------ #
#  BATCH PREPROCESSING                                               #
# ------------------------------------------------------------------ #
def preprocess_all() -> None:
    """
    Preprocesses all images referenced in labels.csv.
    Only processes images that are actually paired with reports.
    """
    print("="*55)
    print("Image Preprocessor")
    print("="*55)

    # -- Load image IDs from labels.csv --
    if not LABELS_CSV.exists():
        print(f"[ERROR] {LABELS_CSV} not found.")
        print("        Run python src/data/parse_reports.py first.")
        return

    image_ids = set()
    with open(LABELS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["image_id"]:
                image_ids.add(row["image_id"])

    print(f"Images to process : {len(image_ids):,}")
    print(f"Output directory  : {OUT_IMAGE_DIR}\n")

    OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    # -- Process each image --
    success = 0
    failed  = []

    for image_id in tqdm(sorted(image_ids), desc="Preprocessing images"):
        raw_path = RAW_IMAGE_DIR / image_id
        out_path = OUT_IMAGE_DIR / image_id

        # Skip if already processed
        if out_path.exists():
            success += 1
            continue

        if not raw_path.exists():
            failed.append(image_id)
            continue

        ok = preprocess_image(raw_path, out_path)
        if ok:
            success += 1
        else:
            failed.append(image_id)

    # -- Summary --
    print(f"\n{'='*55}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*55}")
    print(f"Processed successfully : {success:,}")
    print(f"Failed / not found     : {len(failed):,}")

    if failed:
        print(f"\nFailed images (first 10):")
        for f in failed[:10]:
            print(f"  {f}")

    print(f"\n[OK] Images saved to {OUT_IMAGE_DIR}")
    print("     Next step: python src/data/dataset.py (verify DataLoader)")


# ------------------------------------------------------------------ #
#  VISUAL VERIFICATION (optional)                                    #
# ------------------------------------------------------------------ #
def verify_preprocessing(n_samples: int = 3) -> None:
    """
    Shows before/after for n_samples images.
    Useful to confirm CLAHE is working correctly.
    Run separately: python src/data/preprocess.py --verify
    """
    import matplotlib.pyplot as plt

    raw_images = list(RAW_IMAGE_DIR.glob("*.png"))[:n_samples]
    if not raw_images:
        print("[ERROR] No raw images found for verification.")
        return

    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 4 * n_samples))
    fig.suptitle("CLAHE Preprocessing Verification", fontsize=14)

    for i, raw_path in enumerate(raw_images):
        # Raw
        raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
        axes[i, 0].imshow(raw, cmap="gray")
        axes[i, 0].set_title(f"Raw: {raw_path.name}")
        axes[i, 0].axis("off")

        # Processed
        proc_path = OUT_IMAGE_DIR / raw_path.name
        if proc_path.exists():
            proc = cv2.imread(str(proc_path), cv2.IMREAD_GRAYSCALE)
            axes[i, 1].imshow(proc, cmap="gray")
            axes[i, 1].set_title("After CLAHE + Resize")
            axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/preprocessing_verification.png", dpi=100)
    print("Verification saved → outputs/preprocessing_verification.png")


# ------------------------------------------------------------------ #
#  MAIN                                                              #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys

    if "--verify" in sys.argv:
        verify_preprocessing()
    else:
        preprocess_all()