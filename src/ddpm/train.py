"""
train.py
--------
DDPM training loop.

What this script does:
  1. Loads XRayDataset (real chest X-rays)
  2. Builds UNet + Conditioning + DDPM
  3. Trains DDPM to denoise X-rays (learns to generate them)
  4. Saves best checkpoint
  5. Logs loss to W&B

Run:
  python3 src/ddpm/train.py
  python3 src/ddpm/train.py --epochs 100 --batch_size 16
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ddpm.unet import UNet
from src.ddpm.conditioning import Conditioning
from src.ddpm.diffusion import DDPM
from src.data.dataset import XRayDataset, get_dataloader

# ------------------------------------------------------------------ #
#  CONFIG                                                            #
# ------------------------------------------------------------------ #
def get_config():
    parser = argparse.ArgumentParser(description="Train DDPM on chest X-rays")

    # Model
    parser.add_argument("--base_channels",  type=int,   default=32)
    parser.add_argument("--cond_dim",       type=int,   default=256)
    parser.add_argument("--timesteps",      type=int,   default=1000)
    parser.add_argument("--num_classes",    type=int,   default=18)

    # Training
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=2e-4)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--grad_clip",      type=float, default=1.0)

    # Paths
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints/ddpm")
    parser.add_argument("--log_interval",   type=int,   default=50)
    parser.add_argument("--save_interval",  type=int,   default=10)

    # GPU
    parser.add_argument("--gpu",            type=int,   default=0)
    parser.add_argument("--use_wandb",      action="store_true")

    return parser.parse_args()


# ------------------------------------------------------------------ #
#  TRAINING STEP                                                     #
# ------------------------------------------------------------------ #
def train_one_epoch(
    ddpm      : DDPM,
    loader    : torch.utils.data.DataLoader,
    optimizer : torch.optim.Optimizer,
    device    : torch.device,
    epoch     : int,
    config,
) -> float:
    """
    Runs one full pass through the training data.

    For each batch:
      1. Move images + labels to GPU
      2. Compute DDPM loss (add noise → predict noise → MSE)
      3. Backprop + gradient clip + optimizer step
    """
    ddpm.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        # -- Move to GPU --
        images    = batch["image"].to(device)      # [B, 1, 256, 256]
        class_idx = batch["label"].to(device)      # [B]

# Fix shape if needed
        if images.dim() == 5:
            images = images.squeeze(1)  # [B, 1, 1, 256, 256] → [B, 1, 256, 256]


        # -- Forward pass --
        # DDPM randomly samples timesteps internally
        # Adds noise → UNet predicts noise → MSE loss
        optimizer.zero_grad()
        loss = ddpm.training_loss(images, class_idx)

        # -- Backward pass --
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        # Common in diffusion model training
        torch.nn.utils.clip_grad_norm_(
            ddpm.parameters(),
            max_norm=config.grad_clip
        )

        optimizer.step()

        # -- Logging --
        total_loss  += loss.item()
        num_batches += 1

        if batch_idx % config.log_interval == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches
    return avg_loss


# ------------------------------------------------------------------ #
#  VALIDATION                                                        #
# ------------------------------------------------------------------ #
@torch.no_grad()
def validate(
    ddpm   : DDPM,
    loader : torch.utils.data.DataLoader,
    device : torch.device,
) -> float:
    """
    Computes validation loss.
    Same as training but no gradient computation.
    """
    ddpm.eval()
    total_loss  = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Validating", leave=False):
        images    = batch["image"].to(device)
        class_idx = batch["label"].to(device)

        loss = ddpm.training_loss(images, class_idx)
        total_loss  += loss.item()
        num_batches += 1

    return total_loss / num_batches


# ------------------------------------------------------------------ #
#  SAVE / LOAD CHECKPOINT                                            #
# ------------------------------------------------------------------ #
def save_checkpoint(
    ddpm      : DDPM,
    optimizer : torch.optim.Optimizer,
    epoch     : int,
    loss      : float,
    path      : Path,
) -> None:
    """Saves model weights + optimizer state + epoch info."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch"              : epoch,
        "model_state_dict"   : ddpm.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss"               : loss,
    }, path)
    print(f"[SAVED] Checkpoint → {path}")


def load_checkpoint(
    ddpm      : DDPM,
    optimizer : torch.optim.Optimizer,
    path      : Path,
) -> tuple:
    """Loads checkpoint and returns (start_epoch, best_loss)."""
    if not path.exists():
        print(f"[INFO] No checkpoint found at {path}. Starting fresh.")
        return 0, float("inf")

    checkpoint = torch.load(path, map_location="cpu")
    ddpm.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss  = checkpoint["loss"]
    print(f"[LOADED] Checkpoint from epoch {epoch}, loss {loss:.4f}")
    return epoch + 1, loss


# ------------------------------------------------------------------ #
#  MAIN                                                              #
# ------------------------------------------------------------------ #
def main():
    config = get_config()

    # -- Device setup --
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*55)
    print("DDPM Training")
    print("="*55)
    print(f"Device     : {device}")
    if device.type == "cuda":
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
        print(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Epochs     : {config.epochs}")
    print(f"Batch size : {config.batch_size}")
    print(f"LR         : {config.lr}")
    print()

    # -- W&B logging (optional) --
    if config.use_wandb:
        import wandb
        wandb.init(project="medreportgen", name="ddpm_training", config=vars(config))

    # -- DataLoaders --
    print("Loading datasets...")
    train_loader = get_dataloader(
        XRayDataset,
        split       = "train",
        batch_size  = config.batch_size,
        num_workers = config.num_workers,
    )
    val_loader = get_dataloader(
        XRayDataset,
        split       = "val",
        batch_size  = config.batch_size,
        num_workers = config.num_workers,
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}\n")

    # -- Build model --
    print("Building model...")
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
        unet        = unet,
        conditioning = conditioning,
        timesteps   = config.timesteps,
        device      = str(device),
    ).to(device)

    # Count parameters
    params = sum(p.numel() for p in ddpm.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params:,} ({params/1e6:.1f}M)\n")

    # -- Optimizer + Scheduler --
    optimizer = AdamW(ddpm.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # -- Load checkpoint if exists --
    checkpoint_path = Path(config.checkpoint_dir) / "ddpm_latest.pth"
    best_path       = Path(config.checkpoint_dir) / "ddpm_best.pth"
    start_epoch, best_loss = load_checkpoint(ddpm, optimizer, checkpoint_path)
    # -- Early stopping setup --
    patience         = 5        # stop if no improvement for 5 epochs
    min_delta        = 1e-4     # minimum improvement to count
    patience_counter = 0
    best_loss        = float("inf")

    # -- Training loop --
    print("Starting training...\n")

    for epoch in range(start_epoch, config.epochs):

        # Train
        train_loss = train_one_epoch(
            ddpm, train_loader, optimizer, device, epoch, config
        )

        # Validate
        val_loss = validate(ddpm, val_loader, device)

        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # -- Print progress --
        print(
            f"Epoch {epoch+1:3d}/{config.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # -- W&B logging --
        if config.use_wandb:
            wandb.log({
                "train_loss"     : train_loss,
                "val_loss"       : val_loss,
                "lr"             : current_lr,
                "epoch"          : epoch + 1,
                "patience_counter": patience_counter,
            })

        # -- Save latest checkpoint --
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(ddpm, optimizer, epoch, val_loss, checkpoint_path)

        # -- Early stopping check --
        if val_loss < best_loss - min_delta:
            # Meaningful improvement
            best_loss        = val_loss
            patience_counter = 0
            save_checkpoint(ddpm, optimizer, epoch, val_loss, best_path)
            print(f"  → New best! Val loss: {val_loss:.4f}")
        else:
            # No meaningful improvement
            patience_counter += 1
            print(
                f"  → No improvement. "
                f"Patience: {patience_counter}/{patience} | "
                f"Best: {best_loss:.4f} | "
                f"Current: {val_loss:.4f}"
            )

        # -- Stop if patience exceeded --
        if patience_counter >= patience:
            print(f"\n{'='*55}")
            print(f"[EARLY STOP] Triggered at epoch {epoch+1}")
            print(f"No improvement for {patience} consecutive epochs")
            print(f"Best val loss : {best_loss:.4f}")
            print(f"Best model    : {best_path}")
            print(f"{'='*55}")
            break

    print("\n[DONE] Training complete!")
    print(f"Best val loss : {best_loss:.4f}")
    print(f"Best model    : {best_path}")
    print(f"\nNext step: python3 src/ddpm/sample.py")


if __name__ == "__main__":
    main()