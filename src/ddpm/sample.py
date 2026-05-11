"""
sample.py
---------
Generate synthetic X-rays using trained DDPM.

Purpose:
  Generate synthetic X-rays for RARE disease classes to fix class imbalance.
  Target: bring every class to at least 500 samples.

  fibrosis:      57 real → generate 443 synthetic
  hernia:       103 real → generate 397 synthetic
  emphysema:    242 real → generate 258 synthetic
  calcification: 328 real → generate 172 synthetic
  mass:          367 real → generate 133 synthetic
  fracture:      395 real → generate 105 synthetic

Run:
  python3 src/ddpm/sample.py
  python3 src/ddpm/sample.py --class_name fibrosis --num_samples 443
"""

import os
import sys
import argparse
import torch
import csv
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ddpm.unet import UNet
from src.ddpm.conditioning import Conditioning
from src.ddpm.diffusion import DDPM

# ------------------------------------------------------------------ #
#  CLASS MAP — matches XRayDataset.LABEL_MAP                        #
# ------------------------------------------------------------------ #
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
   

# How many synthetic images to generate per rare class
# Adjust based on your parse_reports.py output
GENERATION_TARGETS = {
    "fibrosis"      : 443,   # 57  real → 500 total
    "hernia"        : 397,   # 103 real → 500 total
    "emphysema"     : 258,   # 242 real → 500 total
    "calcification" : 172,   # 328 real → 500 total
    "mass"          : 133,   # 367 real → 500 total
    "fracture"      : 105,   # 395 real → 500 total
    "nodule"        : 148,   # 352 real → 500 total
    "pneumonia"     : 3,     # 497 real → 500 total
    "edema"         : 112,   # 388 real → 500 total
    "cardiomegaly"  : 141,   # 359 real → 500 total
    "atelectasis"   : 56,    # 444 real → 500 total
    "infiltrate"    : 82,    # 418 real → 500 total
    "opacity"       : 106,   # 394 real → 500 total 500 total
}

# ------------------------------------------------------------------ #
#  LOAD MODEL                                                        #
# ------------------------------------------------------------------ #
def load_ddpm(checkpoint_path: Path, device: torch.device, config) -> DDPM:
    """Loads trained DDPM from checkpoint."""

    unet = UNet(
        in_channels  = 1,
        base_channels = config.base_channels,
        cond_dim     = config.cond_dim,
        num_classes  = config.num_classes,
    ).to(device)

    conditioning = Conditioning(
        num_classes = config.num_classes,
        dim         = config.cond_dim,
    ).to(device)

    ddpm = DDPM(
        unet         = unet,
        conditioning = conditioning,
        timesteps    = config.timesteps,
        device       = str(device),
    ).to(device)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ddpm.load_state_dict(checkpoint["model_state_dict"])
    ddpm.eval()

    print(f"[LOADED] DDPM from {checkpoint_path}")  
    return ddpm


# ------------------------------------------------------------------ #
#  TENSOR → PNG                                                      #
# ------------------------------------------------------------------ #
def tensor_to_png(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts generated tensor → uint8 PNG.

    tensor: [1, H, W] in range [-1, 1]
    Returns: [H, W] uint8 in range [0, 255]
    """
    img = tensor.squeeze(0).cpu().numpy()   # [H, W]
    img = (img + 1) / 2                     # [-1,1] → [0,1]
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


# ------------------------------------------------------------------ #
#  GENERATE FOR ONE CLASS                                            #
# ------------------------------------------------------------------ #
def generate_class(
    ddpm          : DDPM,
    class_name    : str,
    class_idx     : int,
    num_samples   : int,
    out_dir       : Path,
    device        : torch.device,
    batch_size    : int = 4,
    guidance_scale: float = 3.0,
    resume        : bool = False,
) -> list:
    """
    Generates num_samples synthetic X-rays for a specific disease class.
    
    If resume=True, checks how many already exist and continues from there.

    Returns list of saved image paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # Check how many already exist if resuming
    start_idx = 0
    if resume:
        existing = list(out_dir.glob(f"synthetic_{class_name}_*.png"))
        start_idx = len(existing)
        if start_idx > 0:
            print(f"\n[RESUME] {class_name} (class {class_idx}): {start_idx}/{num_samples} already generated")
            if start_idx >= num_samples:
                print(f"[SKIP] {class_name} generation complete ({start_idx} images)")
                return []
        num_samples_to_generate = num_samples - start_idx
    else:
        num_samples_to_generate = num_samples

    print(f"\n[GENERATING] {class_name} (class {class_idx}): {num_samples_to_generate} images (total target: {num_samples})")

    generated = 0
    pbar = tqdm(total=num_samples_to_generate, desc=class_name)

    while generated < num_samples_to_generate:
        # Generate in batches
        current_batch = min(batch_size, num_samples_to_generate - generated)
        class_tensor  = torch.full(
            (current_batch,), class_idx,
            dtype=torch.long, device=device
        )

        # Sample from DDPM
        with torch.no_grad():
            samples = ddpm.sample(
                class_idx      = class_tensor,
                image_size     = 256,
                guidance_scale = guidance_scale,
            )   # [B, 1, 256, 256]

        # Save each generated image
        for i in range(current_batch):
            img = tensor_to_png(samples[i])
            filename = f"synthetic_{class_name}_{start_idx + generated + i:04d}.png"
            save_path = out_dir / filename
            cv2.imwrite(str(save_path), img)
            saved_paths.append(save_path)

        generated += current_batch
        pbar.update(current_batch)

    pbar.close()
    total_now = start_idx + generated
    print(f"[SAVED] {num_samples_to_generate} images → {out_dir} (total: {total_now}/{num_samples})")
    return saved_paths


# ------------------------------------------------------------------ #
#  UPDATE SYNTHETIC LABELS CSV                                       #
# ------------------------------------------------------------------ #
def update_synthetic_labels(
    all_generated: dict,   # class_name → list of paths
    out_csv      : Path,
    resume       : bool = False,
) -> None:
    """
    Creates or updates synthetic_labels.csv with all generated images.

    Format matches labels.csv:
      report_id, image_id, findings, impression, labels, split
      
    If resume=True, appends new images to existing CSV.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    existing_rows = {}
    if resume and out_csv.exists():
        # Load existing rows to avoid duplicates
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                existing_rows[row["image_id"]] = row

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "report_id", "image_id", "findings", "impression", "labels", "split"
        ])
        writer.writeheader()
        
        # Write existing rows first
        for row in existing_rows.values():
            writer.writerow(row)

        # Write newly generated images
        for class_name, paths in all_generated.items():
            for i, path in enumerate(paths):
                if path.name not in existing_rows:
                    writer.writerow({
                        "report_id" : f"synthetic_{class_name}_{i:04d}",
                        "image_id"  : path.name,
                        "findings"  : f"Synthetic {class_name} chest X-ray.",
                        "impression": f"Generated {class_name} finding.",
                        "labels"    : class_name,
                        "split"     : "train",   # synthetic images only in training
                    })

    print(f"\n[SAVED] Synthetic labels → {out_csv}")


# ------------------------------------------------------------------ #
#  MERGE REAL + SYNTHETIC                                            #
# ------------------------------------------------------------------ #
def create_balanced_dataset(
    real_labels_csv      : Path,
    synthetic_labels_csv : Path,
    real_image_dir       : Path,
    synthetic_image_dir  : Path,
    out_dir              : Path,
    out_csv              : Path,
) -> None:
    """
    Combines real + synthetic data into data/balanced/.

    Creates symlinks instead of copying (saves disk space).
    """
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # Load real labels
    with open(real_labels_csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Load synthetic labels
    with open(synthetic_labels_csv) as f:
        for row in csv.DictReader(f):
            # Update image path to point to synthetic dir
            row["image_dir"] = str(synthetic_image_dir)
            rows.append(row)

    # Write merged CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "report_id", "image_id", "findings", "impression", "labels", "split"
        ])
        writer.writeheader()
        for row in rows:
            row.pop("image_dir", None)
            writer.writerow(row)

    print(f"\n[BALANCED DATASET]")
    print(f"Real samples      : {sum(1 for r in rows if 'synthetic' not in r['report_id'])}")
    print(f"Synthetic samples : {sum(1 for r in rows if 'synthetic' in r['report_id'])}")
    print(f"Total             : {len(rows)}")
    print(f"Saved to          : {out_csv}")


# ------------------------------------------------------------------ #
#  MAIN                                                              #
# ------------------------------------------------------------------ #
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     type=str, default="checkpoints/ddpm/ddpm_best.pth")
    parser.add_argument("--class_name",     type=str, default=None,
                        help="Generate for one class only. If None, generate for all rare classes.")
    parser.add_argument("--num_samples",    type=int, default=None,
                        help="Override number of samples to generate.")
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--gpu",            type=int, default=0)
    parser.add_argument("--resume",         action="store_true",
                        help="Resume generation from where it stopped (checks existing files)")

    # Must match train.py config
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--cond_dim",      type=int, default=256)
    parser.add_argument("--num_classes",   type=int, default=14)
    parser.add_argument("--timesteps",     type=int, default=1000)

    return parser.parse_args()


def main():
    config = get_config()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*55)
    print("DDPM Sampling — Synthetic X-ray Generation")
    print("="*55)
    print(f"Device         : {device}")
    print(f"Checkpoint     : {config.checkpoint}")
    print(f"Guidance scale : {config.guidance_scale}")
    print(f"Resume mode    : {'ON' if config.resume else 'OFF'}")
    print()

    # -- Load model --
    checkpoint_path = Path(config.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("        Run python3 src/ddpm/train.py first.")
        return

    ddpm = load_ddpm(checkpoint_path, device, config)

    # -- Determine what to generate --
    if config.class_name:
        # Generate for one specific class
        if config.class_name not in LABEL_MAP:
            print(f"[ERROR] Unknown class: {config.class_name}")
            print(f"        Available: {list(LABEL_MAP.keys())}")
            return
        targets = {
            config.class_name: config.num_samples or GENERATION_TARGETS.get(config.class_name, 100)
        }
    else:
        # Generate for all rare classes
        targets = GENERATION_TARGETS

    # -- Generate --
    out_dir      = Path("data/synthetic/ddpm_generated")
    all_generated = {}

    for class_name, num_samples in targets.items():
        if class_name not in LABEL_MAP:
            print(f"[SKIP] {class_name} not in LABEL_MAP")
            continue

        class_idx  = LABEL_MAP[class_name]
        class_dir  = out_dir / class_name

        paths = generate_class(
            ddpm           = ddpm,
            class_name     = class_name,
            class_idx      = class_idx,
            num_samples    = num_samples,
            out_dir        = class_dir,
            device         = device,
            batch_size     = config.batch_size,
            guidance_scale = config.guidance_scale,
            resume         = config.resume,
        )
        all_generated[class_name] = paths

    # -- Save synthetic labels CSV --
    synthetic_csv = Path("data/synthetic/synthetic_labels.csv")
    update_synthetic_labels(all_generated, synthetic_csv, resume=config.resume)

    # -- Create balanced dataset --
    create_balanced_dataset(
        real_labels_csv      = Path("data/processed/labels.csv"),
        synthetic_labels_csv = synthetic_csv,
        real_image_dir       = Path("data/processed/images"),
        synthetic_image_dir  = out_dir,
        out_dir              = Path("data/balanced/images"),
        out_csv              = Path("data/balanced/final_labels.csv"),
    )

    print("\n[DONE] Synthetic generation complete!")
    print("Next step: python3 src/ddpm/evaluate.py")


if __name__ == "__main__":
    main()