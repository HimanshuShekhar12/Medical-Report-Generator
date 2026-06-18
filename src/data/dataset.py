"""
dataset.py
----------
PyTorch Dataset classes for MedReportGen.

Three dataset classes:
  1. XRayDataset      → image only (used by DDPM + VAE training)
  2. ReportDataset    → text only  (used for tokenizer analysis)
  3. XRayReportDataset → image + report pairs (used by BioGPT training)

Usage:
  from src.data.dataset import XRayReportDataset
  dataset = XRayReportDataset(split="train")
"""

import csv
import torch
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BioGptTokenizer

# ------------------------------------------------------------------ #
#  PATHS                                                             #
# ------------------------------------------------------------------ #
PROCESSED_IMAGE_DIR = Path("data/processed/images")
BALANCED_IMAGE_DIR  = Path("data/balanced/images")       # real + synthetic
LABELS_CSV          = Path("data/processed/labels.csv")
FINAL_LABELS_CSV    = Path("data/balanced/final_labels.csv")

# ------------------------------------------------------------------ #
#  CONFIG                                                            #
# ------------------------------------------------------------------ #
IMAGE_SIZE  = 256
MAX_TEXT_LEN = 128      # max tokens for report text


# ------------------------------------------------------------------ #
#  DATASET 1: XRayDataset (image only — for DDPM and VAE)           #
# ------------------------------------------------------------------ #
class XRayDataset(Dataset):
    """
    Returns preprocessed chest X-ray images as tensors.
    Used for:
      - DDPM training (generate synthetic images)
      - VAE training  (learn image latent representations)

    Returns:
      image : torch.Tensor [1, 256, 256]  (grayscale, normalized 0-1)
      label : int  (pathology class index, used by conditional DDPM)
    """

    # Map pathology name → integer class index
    # 18 classes total
    LABEL_MAP = {
        "normal"        : 0,
        "effusion"      : 1,
        "pleural"       : 2,
        "pneumothorax"  : 3,
        "consolidation" : 4,
        "infiltrate"    : 5,
        "opacity"       : 6,
        "atelectasis"   : 7,
        "edema"         : 8,
        "cardiomegaly"  : 9,
        "nodule"        : 10,
        "pneumonia"     : 11,
        "fracture"      : 12,
        "mass"          : 13,
        "calcification" : 14,
        "emphysema"     : 15,
        "hernia"        : 16,
        "fibrosis"      : 17,
    }

    NUM_CLASSES = len(LABEL_MAP)  # 18
        
  

    def __init__(
        self,
        split          : str  = "train",   # "train", "val", "test"
        use_balanced   : bool = False,     # True = use real + synthetic
        image_size     : int  = IMAGE_SIZE,
        augment        : bool = False,     # flip augmentation
    ):
        self.split       = split
        self.image_size  = image_size
        self.augment     = augment

        # Use balanced dataset (real + synthetic) or processed only
        if use_balanced and FINAL_LABELS_CSV.exists():
            labels_csv  = FINAL_LABELS_CSV
            self.img_dir = BALANCED_IMAGE_DIR
        else:
            labels_csv  = LABELS_CSV
            self.img_dir = PROCESSED_IMAGE_DIR

        # Load rows for this split
        self.samples = self._load_samples(labels_csv, split)
        print(f"[XRayDataset] {split}: {len(self.samples):,} samples")

    def _load_samples(self, csv_path: Path, split: str) -> list:
        samples = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                if not row["image_id"]:
                    continue

                img_path = self.img_dir / row["image_id"]
                if not img_path.exists():
                    continue

                # Get ALL labels (not just first one)
                raw_labels = row["labels"].split("|")
                valid_labels = [l for l in raw_labels if l in self.LABEL_MAP]

                if not valid_labels:
                    continue

                # Keep primary label for DDPM conditioning (backward compatible)
                label_idx = self.LABEL_MAP[valid_labels[0]]

                # Multi-hot vector for classifier training
                multi_hot = [0] * self.NUM_CLASSES
                for l in valid_labels:
                    multi_hot[self.LABEL_MAP[l]] = 1

                samples.append({
                    "image_path": img_path,
                    "label"     : label_idx,      # used by DDPM (single class)
                    "multi_hot" : multi_hot,       # used by classifier (all classes)
                    "report_id" : row["report_id"],
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # -- Load image --
        img = cv2.imread(str(sample["image_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Return blank image if file is corrupt
            img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # -- Resize if needed --
        if img.shape[0] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))

        # -- Augmentation (horizontal flip only for X-rays) --
        if self.augment and torch.rand(1).item() > 0.5:
            img = cv2.flip(img, 1)

        # -- Normalize to [0, 1] and convert to tensor --
        # Shape: [1, H, W] — single channel (grayscale)
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor * 2.0 - 1.0  # [0,1] → [-1,1]
        img_tensor = img_tensor.unsqueeze(0)  # add channel dim

        return {
            "image"     : img_tensor,                    # [1, 256, 256]
            "label"     : sample["label"],                # int (primary, for DDPM)
            "multi_hot" : torch.tensor(sample["multi_hot"], dtype=torch.float32),  # [18]
            "report_id" : sample["report_id"],            # str
        }


# ------------------------------------------------------------------ #
#  DATASET 2: XRayReportDataset (image + text — for BioGPT)         #
# ------------------------------------------------------------------ #
class XRayReportDataset(Dataset):
    """
    Returns (image tensor, tokenized report) pairs.
    Used for:
      - BioGPT fine-tuning (Stage 3 training)

    The report text format fed to BioGPT:
      "FINDINGS: {findings} IMPRESSION: {impression}"

    Returns:
      image      : torch.Tensor [1, 256, 256]
      input_ids  : torch.Tensor [MAX_TEXT_LEN]   (tokenized report)
      labels     : torch.Tensor [MAX_TEXT_LEN]   (same as input_ids for LM)
      label_idx  : int (pathology class)
    """

    def __init__(
        self,
        split        : str  = "train",
        use_balanced : bool = False,
        max_length   : int  = MAX_TEXT_LEN,
        augment      : bool = False,
    ):
        self.max_length = max_length
        self.augment    = augment

        # Load base XRayDataset for image handling
        self._img_dataset = XRayDataset(
            split=split,
            use_balanced=use_balanced,
            augment=augment,
        )

        # Load BioGPT tokenizer
        print("[XRayReportDataset] Loading BioGPT tokenizer...")
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load report texts
        labels_csv = FINAL_LABELS_CSV if use_balanced else LABELS_CSV
        self.report_map = self._load_reports(labels_csv, split)

        # Filter samples that have both image and report
        self.samples = [
            s for s in self._img_dataset.samples
            if s["report_id"] in self.report_map
        ]
        print(f"[XRayReportDataset] {split}: {len(self.samples):,} paired samples")

    def _load_reports(self, csv_path: Path, split: str) -> dict:
        """Returns dict: report_id → formatted report text"""
        report_map = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                report_id = row["report_id"]
                if report_id in report_map:
                    continue  # already loaded

                # Format report for BioGPT
                text = format_report_text(
                    findings   = row["findings"],
                    impression = row["impression"],
                )
                report_map[report_id] = text
        return report_map

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # -- Load image (reuse XRayDataset logic) --
        img_item = self._img_dataset.__getitem__(
            self._img_dataset.samples.index(sample)
        )

        # -- Tokenize report --
        report_text = self.report_map[sample["report_id"]]
        encoded = self.tokenizer(
            report_text,
            max_length  = self.max_length,
            truncation  = True,
            padding     = "max_length",
            return_tensors = "pt",
        )

        input_ids = encoded["input_ids"].squeeze()   # [MAX_TEXT_LEN]

        return {
            "image"    : img_item["image"],   # [1, 256, 256]
            "input_ids": input_ids,            # [MAX_TEXT_LEN]
            "labels"   : input_ids.clone(),    # same for LM training
            "label_idx": img_item["label"],    # int
        }


# ------------------------------------------------------------------ #
#  HELPER: Format report text for BioGPT                            #
# ------------------------------------------------------------------ #
def format_report_text(findings: str, impression: str) -> str:
    """
    Formats findings + impression into a single string for BioGPT.

    Format:
      "FINDINGS: <findings text> IMPRESSION: <impression text>"

    This structured format helps BioGPT learn to generate
    both sections in the correct order.
    """
    parts = []
    if findings:
        parts.append(f"FINDINGS: {findings}")
    if impression:
        parts.append(f"IMPRESSION: {impression}")
    return " ".join(parts)


# ------------------------------------------------------------------ #
#  DATALOADER FACTORY                                                #
# ------------------------------------------------------------------ #
def get_dataloader(
    dataset_class,
    split       : str  = "train",
    batch_size  : int  = 16,
    num_workers : int  = 4,
    use_balanced: bool = False,
    **kwargs,
) -> DataLoader:
    """
    Creates a DataLoader for the given dataset class and split.

    Example:
        from src.data.dataset import XRayDataset, get_dataloader
        loader = get_dataloader(XRayDataset, split="train", batch_size=32)
    """
    dataset = dataset_class(
        split=split,
        use_balanced=use_balanced,
        augment=(split == "train"),
        **kwargs,
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (split == "train"),
        num_workers = num_workers,
        pin_memory  = True,    # faster GPU transfer
        drop_last   = (split == "train"),
    )

    return loader


# ------------------------------------------------------------------ #
#  QUICK VERIFICATION — run this file directly to test               #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("="*55)
    print("Dataset Verification")
    print("="*55)

    # Test XRayDataset
    print("\n[1] Testing XRayDataset...")
    try:
        ds = XRayDataset(split="train")
        if len(ds) > 0:
            sample = ds[0]
            print(f"    Image shape : {sample['image'].shape}")
            print(f"    Label       : {sample['label']}")
            print(f"    Image range : [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
            print(f"    [OK] XRayDataset working")
        else:
            print("    [WARNING] Dataset is empty — run download + preprocess first")
    except Exception as e:
        print(f"    [ERROR] {e}")

    # Test DataLoader
    print("\n[2] Testing DataLoader (batch_size=4)...")
    try:
        loader = get_dataloader(XRayDataset, split="train", batch_size=4, num_workers=0)
        batch  = next(iter(loader))
        print(f"    Batch image shape : {batch['image'].shape}")
        print(f"    Batch labels      : {batch['label']}")
        print(f"    [OK] DataLoader working")
    except Exception as e:
        print(f"    [ERROR] {e}")

    print("\nNext step: python src/ddpm/train.py")