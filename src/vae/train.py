"""
train.py
--------
VAE Training loop for chest X-ray images.

Run:
  python3 src/vae/train.py
"""

import os
import sys
import argparse
import torch
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.vae.vae import VAE
from src.data.dataset import XRayDataset, get_dataloader


def get_config():
    parser = argparse.ArgumentParser(description="Train VAE on X-Ray Dataset")
    parser.add_argument("--latent_channels", type=int,   default=256)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--num_workers",     type=int,   default=4)
    parser.add_argument("--checkpoint_dir",  type=str,   default="checkpoints/vae")
    parser.add_argument("--save_interval",   type=int,   default=10)
    parser.add_argument("--patience",        type=int,   default=5)
    parser.add_argument("--gpu",             type=int,   default=1)
    return parser.parse_args()


def train_one_epoch(vae, loader, optimizer, device):
    vae.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch["image"].to(device)   # [B, 1, 256, 256]

        optimizer.zero_grad()
        x_recon, mu, logvar = vae(x)
        loss, recon_loss, kl_loss = vae.loss_function(x_recon, x, mu, logvar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item()
        total_recon += recon_loss.item()
        total_kl    += kl_loss.item()

    n = len(loader)
    return total_loss/n, total_recon/n, total_kl/n


@torch.no_grad()
def validate(vae, loader, device):
    vae.eval()
    total_loss = 0

    for batch in tqdm(loader, desc="Validating", leave=False):
        x = batch["image"].to(device)
        x_recon, mu, logvar = vae(x)
        loss, _, _ = vae.loss_function(x_recon, x, mu, logvar)
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    config = get_config()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*55)
    print("VAE Training")
    print("="*55)
    print(f"Device     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Epochs     : {config.epochs}")
    print(f"Batch size : {config.batch_size}")
    print(f"LR         : {config.lr}\n")

    # DataLoaders
    print("Loading datasets...")
    train_loader = get_dataloader(
        XRayDataset,
        split        = "train",
        batch_size   = config.batch_size,
        num_workers  = config.num_workers,
        use_balanced = True,    # real + synthetic
    )
    val_loader = get_dataloader(
        XRayDataset,
        split        = "val",
        batch_size   = config.batch_size,
        num_workers  = config.num_workers,
        use_balanced = False,   # real only for validation
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}\n")

    # Model
    vae = VAE(in_channels=1, latent_channels=config.latent_channels).to(device)
    params = sum(p.numel() for p in vae.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)\n")

    # Optimizer
    optimizer = AdamW(vae.parameters(), lr=config.lr, weight_decay=1e-4)

    # Checkpoint paths
    ckpt_dir  = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path   = ckpt_dir / "vae_best.pth"
    latest_path = ckpt_dir / "vae_latest.pth"

    # Early stopping
    best_loss        = float("inf")
    patience_counter = 0
    min_delta        = 1e-4

    print("Starting training...\n")

    for epoch in range(config.epochs):

        # Train
        train_loss, recon_loss, kl_loss = train_one_epoch(
            vae, train_loader, optimizer, device
        )

        # Validate
        val_loss = validate(vae, val_loader, device)

        # Print progress
        print(
            f"Epoch {epoch+1:3d}/{config.epochs} | "
            f"Train: {train_loss:.4f} "
            f"(Recon: {recon_loss:.4f} KL: {kl_loss:.4f}) | "
            f"Val: {val_loss:.4f}"
        )

        # Save latest
        if (epoch + 1) % config.save_interval == 0:
            torch.save({
                "epoch"               : epoch,
                "model_state_dict"    : vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss"                : val_loss,
            }, latest_path)

        # Early stopping + best model
        if val_loss < best_loss - min_delta:
            best_loss        = val_loss
            patience_counter = 0
            torch.save({
                "epoch"               : epoch,
                "model_state_dict"    : vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss"                : val_loss,
            }, best_path)
            print(f"  → New best! Val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  → No improvement. Patience: {patience_counter}/{config.patience}")

        if patience_counter >= config.patience:
            print(f"\n[EARLY STOP] Triggered at epoch {epoch+1}")
            print(f"Best val loss: {best_loss:.4f}")
            break

    print(f"\n[DONE] Training complete!")
    print(f"Best val loss : {best_loss:.4f}")
    print(f"Best model    : {best_path}")


if __name__ == "__main__":
    main()
    