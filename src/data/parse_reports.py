"""
parse_reports.py
----------------
Parses OpenI XML report files into clean (findings, impression) text pairs.

OpenI XML structure (what we extract):
  <MedlineCitation>
    <Article>
      <Abstract>
        <AbstractText Label="FINDINGS">  ← we extract this
        <AbstractText Label="IMPRESSION"> ← and this
      </Abstract>
    </Article>
  </MedlineCitation>

Output:
  data/processed/reports/   → one .txt file per report
  data/processed/labels.csv → image_id | findings | impression | split

Usage:
  python src/data/parse_reports.py
"""

import os
import re
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import random

# ------------------------------------------------------------------ #
#  PATHS                                                             #
# ------------------------------------------------------------------ #
REPORT_DIR     = Path("data/raw/openI/reports")
IMAGE_DIR      = Path("data/raw/openI/images")
PROCESSED_DIR  = Path("data/processed")
OUT_REPORT_DIR = PROCESSED_DIR / "reports"
LABELS_CSV     = PROCESSED_DIR / "labels.csv"

# Train / Val / Test split ratios
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

# ------------------------------------------------------------------ #
#  KNOWN PATHOLOGY LABELS IN OPENI                                   #
# ------------------------------------------------------------------ #
# These are the 18 pathologies OpenI annotates.
# We'll detect them in the text to create weak labels.
PATHOLOGIES = [
    "normal", "cardiomegaly", "effusion", "infiltrate",
    "pneumonia", "atelectasis", "pneumothorax", "edema",
    "consolidation", "pleural", "mass", "nodule",
    "hernia", "emphysema", "fibrosis", "fracture",
    "calcification", "opacity",
]


# ------------------------------------------------------------------ #
#  XML PARSING                                                       #
# ------------------------------------------------------------------ #
def parse_single_report(xml_path: Path) -> dict:
    """
    Parses one OpenI XML file.

    Returns a dict:
    {
        "report_id"  : "1234",
        "image_ids"  : ["CXR1234_IM-0001-1001.png", ...],
        "findings"   : "The chest X-ray shows ...",
        "impression" : "No acute findings.",
        "labels"     : ["cardiomegaly", "effusion"],   # weak labels from text
    }
    Returns None if report has no usable text.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return None

    report_id = xml_path.stem  # filename without .xml

    # -- Extract findings and impression --
    findings   = ""
    impression = ""

    # OpenI uses AbstractText with Label attribute
    for abstract_text in root.iter("AbstractText"):
        label = abstract_text.get("Label", "").upper()
        text  = (abstract_text.text or "").strip()

        if label == "FINDINGS":
            findings = text
        elif label == "IMPRESSION":
            impression = text

    # Skip reports with no useful text
    if not findings and not impression:
        return None

    # -- Extract linked image filenames --
    image_ids = []
    for fig in root.iter("parentImage"):
        img_id = fig.get("id", "")
        if img_id:
            # OpenI stores image IDs like "CXR1234_IM-0001-1001"
            # The actual file is CXR1234_IM-0001-1001.png
            image_ids.append(img_id + ".png")

    # -- Detect pathology labels from text --
    combined_text = (findings + " " + impression).lower()
    labels = []
    for pathology in PATHOLOGIES:
        if pathology in combined_text:
            labels.append(pathology)

    # If no pathology found → "normal"
    if not labels:
        labels = ["normal"]

    return {
        "report_id" : report_id,
        "image_ids" : image_ids,
        "findings"  : clean_text(findings),
        "impression": clean_text(impression),
        "labels"    : labels,
    }


def clean_text(text: str) -> str:
    """
    Cleans raw report text:
    - Removes extra whitespace
    - Removes non-printable characters
    - Fixes common XML artifacts
    - Normalizes to lowercase (optional — keep False for generation)
    """
    if not text:
        return ""

    # Remove XML/HTML artifacts
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Replace common abbreviations for clarity
    # (helps BioGPT tokenize better)
    text = text.replace("w/", "with")
    text = text.replace("w/o", "without")
    text = text.replace("bilat.", "bilateral")
    text = text.replace("bilat", "bilateral")
    text = text.replace("r/o", "rule out")
    text = text.replace("s/p", "status post")

    return text


# ------------------------------------------------------------------ #
#  DATASET SPLIT                                                     #
# ------------------------------------------------------------------ #
def assign_split(idx: int, total: int) -> str:
    """
    Assigns train/val/test split based on index.
    Deterministic — same split every run.
    """
    ratio = idx / total
    if ratio < SPLIT_RATIOS["train"]:
        return "train"
    elif ratio < SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]:
        return "val"
    else:
        return "test"


# ------------------------------------------------------------------ #
#  MAIN                                                              #
# ------------------------------------------------------------------ #
def main():
    print("="*55)
    print("OpenI Report Parser")
    print("="*55)

    # -- Setup output directories --
    OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # -- Find all XML files --
    xml_files = sorted(REPORT_DIR.rglob("*.xml"))
    if not xml_files:
        print(f"[ERROR] No XML files found in {REPORT_DIR}")
        print("        Run python src/data/download.py first.")
        return

    print(f"Found {len(xml_files):,} XML report files\n")

    # -- Shuffle for random split (fixed seed for reproducibility) --
    random.seed(42)
    random.shuffle(xml_files)

    # -- Parse all reports --
    parsed     = []
    skipped    = 0

    for xml_path in tqdm(xml_files, desc="Parsing XML reports"):
        result = parse_single_report(xml_path)
        if result is None:
            skipped += 1
            continue
        parsed.append(result)

    print(f"\nParsed  : {len(parsed):,} reports")
    print(f"Skipped : {skipped:,} (empty/corrupt)")

    # -- Save individual .txt files --
    print("\nSaving individual report text files...")
    for item in tqdm(parsed, desc="Writing .txt files"):
        txt_path = OUT_REPORT_DIR / f"{item['report_id']}.txt"
        with open(txt_path, "w") as f:
            f.write(f"FINDINGS: {item['findings']}\n")
            f.write(f"IMPRESSION: {item['impression']}\n")

    # -- Save labels.csv --
    print(f"\nSaving labels CSV → {LABELS_CSV}")

    with open(LABELS_CSV, "w", newline="") as csvfile:
        fieldnames = [
            "report_id", "image_id", "findings",
            "impression", "labels", "split"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total = len(parsed)
        for idx, item in enumerate(parsed):
            split = assign_split(idx, total)

            # One row per image (a report can have frontal + lateral)
            image_ids = item["image_ids"] if item["image_ids"] else [""]
            for image_id in image_ids:
                writer.writerow({
                    "report_id" : item["report_id"],
                    "image_id"  : image_id,
                    "findings"  : item["findings"],
                    "impression": item["impression"],
                    "labels"    : "|".join(item["labels"]),
                    "split"     : split,
                })

    # -- Print dataset summary --
    print_summary(parsed)


def print_summary(parsed: list) -> None:
    """Prints class distribution and split counts."""

    from collections import Counter

    all_labels = []
    for item in parsed:
        all_labels.extend(item["labels"])

    label_counts = Counter(all_labels)

    print("\n" + "="*55)
    print("DATASET SUMMARY")
    print("="*55)
    print(f"Total usable reports : {len(parsed):,}")

    # Split counts
    total = len(parsed)
    train = int(total * SPLIT_RATIOS["train"])
    val   = int(total * SPLIT_RATIOS["val"])
    test  = total - train - val

    print(f"\nSplit:")
    print(f"  Train : {train:,}")
    print(f"  Val   : {val:,}")
    print(f"  Test  : {test:,}")

    print(f"\nPathology distribution:")
    for label, count in label_counts.most_common():
        bar = "█" * (count // 20)
        print(f"  {label:<20} {count:>5}  {bar}")

    # Identify rare classes (< 100 samples)
    rare = [l for l, c in label_counts.items() if c < 100]
    if rare:
        print(f"\n[!] Rare classes (< 100 samples) → DDPM will augment these:")
        for r in rare:
            print(f"    - {r}: {label_counts[r]} samples")

    print("\n[OK] Parsing complete.")
    print("     Next step: python src/data/preprocess.py")


if __name__ == "__main__":
    main()
