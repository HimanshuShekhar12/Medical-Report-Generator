"""
generate.py
-----------
Inference: chest X-ray image → generated radiology report.

Run on a single image:
  python3 src/report_gen/generate.py --image_path data/processed/images/CXR1000_IM-0003-1001.png

Run on the test set (batch evaluation):
  python3 src/report_gen/generate.py --eval_test
"""

import sys
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.report_gen.model import MedReportGenerator
from src.data.dataset import XRayReportDataset, get_dataloader


def get_config():
    parser = argparse.ArgumentParser(description="Generate reports from X-rays")
    parser.add_argument("--vae_checkpoint",      type=str, default="checkpoints/vae/vae_best.pth")
    parser.add_argument("--biogpt_checkpoint",   type=str, default="checkpoints/report_gen/biogpt_best.pth")
    parser.add_argument("--num_tokens",          type=int, default=16)
    parser.add_argument("--image_path",          type=str, default=None)
    parser.add_argument("--eval_test",           action="store_true")
    parser.add_argument("--max_length",          type=int, default=128)
    parser.add_argument("--num_beams",           type=int, default=4)
    parser.add_argument("--gpu",                 type=int, default=1)
    return parser.parse_args()


def load_model(config, device):
    model = MedReportGenerator(
        vae_checkpoint = config.vae_checkpoint,
        num_tokens     = config.num_tokens,
        freeze_vae     = True,
    ).to(device)

    ckpt = torch.load(config.biogpt_checkpoint, map_location=device)
    model.projection.load_state_dict(ckpt["projection_state_dict"])
    model.biogpt.load_state_dict(ckpt["biogpt_state_dict"])
    model.eval()
    return model


def load_image_as_tensor(image_path: str, image_size: int = 256) -> torch.Tensor:
    """Loads a single PNG and prepares it the same way XRayDataset does."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if img.shape[0] != image_size:
        img = cv2.resize(img, (image_size, image_size))

    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor * 2.0 - 1.0          # [0,1] → [-1,1]
    img_tensor = img_tensor.unsqueeze(0)          # add channel dim [1,H,W]
    img_tensor = img_tensor.unsqueeze(0)          # add batch dim [1,1,H,W]
    return img_tensor


def generate_single(model, image_path, device, config):
    image = load_image_as_tensor(image_path).to(device)
    report = model.generate(
        image,
        max_length = config.max_length,
        num_beams  = config.num_beams,
    )
    return report


def evaluate_on_test_set(model, device, config):
    """
    Generates reports for the test split and prints them alongside
    ground truth — quick qualitative check before computing
    BLEU/ROUGE metrics properly.
    """
    test_loader = get_dataloader(
        XRayReportDataset,
        split        = "test",
        batch_size   = 1,
        num_workers  = 0,
        use_balanced = False,
    )

    print(f"Generating reports for {len(test_loader)} test samples...\n")

    results = []
    for i, batch in enumerate(test_loader):
        image = batch["image"].to(device)

        generated = model.generate(
            image,
            max_length = config.max_length,
            num_beams  = config.num_beams,
        )

        ground_truth = model.tokenizer.decode(
            batch["input_ids"][0], skip_special_tokens=True
        )

        results.append({"generated": generated, "ground_truth": ground_truth})

        print(f"--- Sample {i+1} ---")
        print(f"GROUND TRUTH : {ground_truth[:200]}")
        print(f"GENERATED    : {generated[:200]}")
        print()

        if i >= 9:   # just show first 10 for a quick look
            break

    return results


def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model on {device}...")
    model = load_model(config, device)
    print("[OK] Model loaded\n")

    if config.eval_test:
        evaluate_on_test_set(model, device, config)
    elif config.image_path:
        report = generate_single(model, config.image_path, device, config)
        print(f"\nGenerated Report:\n{report}")
    else:
        print("[ERROR] Provide either --image_path or --eval_test")


if __name__ == "__main__":
    main()