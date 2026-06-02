# Evaluation for VAE
"""
evaluation.py
-------------
VAE evaluation tools for chest X-ray reconstruction.

This module loads a trained VAE checkpoint, runs the model on an evaluation
split, reports average reconstruction / KL / total loss, and optionally saves
reconstruction examples.
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.vae.vae import VAE
from src.data.dataset import XRayDataset, get_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained VAE on X-ray dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vae/vae_best.pth",
                        help="Path to the VAE checkpoint")
    parser.add_argument("--latent_channels", type=int, default=256,
                        help="Number of latent channels in the VAE")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--output_dir", type=str, default="outputs/vae_eval",
                        help="Directory to save reconstructions")
    parser.add_argument("--save_samples", action="store_true",
                        help="Save example reconstructions to disk")
    parser.add_argument("--num_samples", type=int, default=16,
                        help="Maximum number of example reconstructions to save")
    parser.add_argument("--save_interval", type=int, default=8,
                        help="Save every Nth sample from the evaluation set")
    return parser.parse_args()


def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    """Convert image tensor from [-1, 1] to [0, 1]."""
    return img_tensor.clamp(-1.0, 1.0).add(1.0).div(2.0)


def psnr(target: torch.Tensor, prediction: torch.Tensor, data_range: float = 1.0) -> float:
    mse = torch.mean((target - prediction) ** 2)
    if mse.item() == 0.0:
        return float("inf")
    return 10.0 * torch.log10((data_range ** 2) / mse).item()


def save_reconstructions(x: torch.Tensor, x_recon: torch.Tensor, output_dir: Path, start_idx: int = 0):
    """Save side-by-side original / reconstructed images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    x = denormalize(x).cpu().numpy()
    x_recon = x_recon.cpu().numpy()

    num_images = min(x.shape[0], x_recon.shape[0])
    for idx in range(num_images):
        orig = (x[idx, 0] * 255.0).clip(0, 255).astype(np.uint8)
        recon = (x_recon[idx, 0] * 255.0).clip(0, 255).astype(np.uint8)
        combined = np.concatenate([orig, recon], axis=1)
        filename = output_dir / f"recon_{start_idx + idx:04d}.png"
        cv2.imwrite(str(filename), combined)


def evaluate(vae: VAE, loader, device: torch.device, save_dir: Path = None, num_samples: int = 16, save_interval: int = 8):
    vae.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_psnr = 0.0
    sample_count = 0
    saved_count = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            x = batch["image"].to(device)
            x_recon, mu, logvar = vae(x)

            # Align the input range with the decoder output range [0,1]
            x_target = denormalize(x)
            loss, recon_loss, kl_loss = vae.loss_function(x_recon, x_target, mu, logvar)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_psnr += psnr(x_target, x_recon)
            sample_count += 1

            if save_dir is not None and saved_count < num_samples:
                batch_save = min(num_samples - saved_count, x.shape[0])
                save_reconstructions(
                    x[:batch_save],
                    x_recon[:batch_save],
                    save_dir,
                    start_idx=saved_count,
                )
                saved_count += batch_save

    if sample_count == 0:
        raise RuntimeError("No examples were evaluated. Check the dataset split and paths.")

    return {
        "loss": total_loss / sample_count,
        "recon_loss": total_recon / sample_count,
        "kl_loss": total_kl / sample_count,
        "psnr": total_psnr / sample_count,
    }


def load_checkpoint(vae: VAE, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    vae.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def main():
    config = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("VAE Evaluation")
    print("=" * 60)
    print(f"Device        : {device}")
    if device.type == "cuda":
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint    : {config.checkpoint}")
    print(f"Dataset split : {config.split}")
    print(f"Batch size    : {config.batch_size}")
    print(f"Output dir    : {config.output_dir}")
    print("=" * 60)

    loader = get_dataloader(
        XRayDataset,
        split=config.split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_balanced=False,
    )

    vae = VAE(in_channels=1, latent_channels=config.latent_channels).to(device)
    load_checkpoint(vae, Path(config.checkpoint), device)

    save_dir = Path(config.output_dir) if config.save_samples else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate(
        vae,
        loader,
        device,
        save_dir=save_dir,
        num_samples=config.num_samples,
        save_interval=config.save_interval,
    )

    print("Evaluation results:")
    print(f"  Total loss : {metrics['loss']:.6f}")
    print(f"  Recon loss : {metrics['recon_loss']:.6f}")
    print(f"  KL loss    : {metrics['kl_loss']:.6f}")
    print(f"  PSNR       : {metrics['psnr']:.2f} dB")

    if save_dir is not None:
        print(f"Saved example reconstructions to: {save_dir}")


if __name__ == "__main__":
    main()
