"""
train.py
--------
Trains CheXNet multi-label classifier on chest X-ray data.

Supports two modes (for ablation study):
  --use_balanced False  → real images only (baseline)
  --use_balanced True   → real + DDPM synthetic images (augmented)

Run baseline:
  python3 src/classifier/train.py --use_balanced false --tag baseline

Run augmented:
  python3 src/classifier/train.py --use_balanced true --tag augmented

Compare F1 scores between the two runs → ablation study evidence.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.classifier.chexnet import CheXNet
from src.data.dataset import XRayDataset, get_dataloader


def get_config():
    parser = argparse.ArgumentParser(description="Train CheXNet classifier")
    parser.add_argument("--num_classes",  type=int,   default=18)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--use_balanced", type=str,   default="true")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/classifier")
    parser.add_argument("--tag",          type=str,   default="augmented")
    parser.add_argument("--patience",     type=int,   default=10)
    parser.add_argument("--gpu",          type=int,   default=1)
    args = parser.parse_args()
    args.use_balanced = args.use_balanced.lower() == "true"
    return args

def train_one_epoch(model, loader, optimizer, criterion, device, num_classes):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        x = batch["image"].to(device)
        y = batch["multi_hot"].to(device)

        optimizer.zero_grad()
        probs = model(x)
        loss  = criterion(probs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0

    for batch in tqdm(loader, desc="Validating", leave=False):
        x = batch["image"].to(device)
        y = batch["multi_hot"].to(device)

        probs = model(x)
        loss  = criterion(probs, y)
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    config = get_config()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*55)
    print("CheXNet Classifier Training")
    print("="*55)
    print(f"Device       : {device}")
    print(f"Tag          : {config.tag}")
    print(f"Use balanced : {config.use_balanced}")
    print(f"Epochs       : {config.epochs}")
    print(f"Batch size   : {config.batch_size}\n")

    print("Loading datasets...")
    train_loader = get_dataloader(
        XRayDataset,
        split        = "train",
        batch_size   = config.batch_size,
        num_workers  = config.num_workers,
        use_balanced = config.use_balanced,
    )
    val_loader = get_dataloader(
        XRayDataset,
        split        = "val",
        batch_size   = config.batch_size,
        num_workers  = config.num_workers,
        use_balanced = False,   # always evaluate on real data only
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}\n")

    model = CheXNet(num_classes=config.num_classes).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)\n")

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    criterion = nn.BCELoss()   # model already applies Sigmoid internally

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"chexnet_{config.tag}_best.pth"

    best_loss        = float("inf")
    patience_counter = 0

    print("Starting training...\n")

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config.num_classes)
        val_loss   = validate(model, val_loader, criterion, device, config.num_classes)

        print(f"Epoch {epoch+1:3d}/{config.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_loss - 1e-4:
            best_loss        = val_loss
            patience_counter  = 0
            torch.save({
                "epoch"            : epoch,
                "model_state_dict" : model.state_dict(),
                "loss"             : val_loss,
                "tag"              : config.tag,
            }, best_path)
            print(f"  → New best! Val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  → No improvement. Patience: {patience_counter}/{config.patience}")

        if patience_counter >= config.patience:
            print(f"\n[EARLY STOP] Triggered at epoch {epoch+1}")
            break

    print(f"\n[DONE] Training complete!")
    print(f"Best val loss : {best_loss:.4f}")
    print(f"Best model    : {best_path}")
    print(f"Next: python3 src/classifier/evaluate.py --tag {config.tag}")


if __name__ == "__main__":
    main()