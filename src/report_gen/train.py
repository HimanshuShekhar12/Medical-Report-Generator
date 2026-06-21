"""
train.py
--------
Fine-tunes MedReportGenerator (VAE + Projection + BioGPT) on
(image, report) pairs from the OpenI dataset.

What actually gets trained:
  VAE          → FROZEN (already trained in Phase 3)
  Projection   → TRAINABLE (new layer, ~1M params)
  BioGPT       → TRAINABLE (fine-tuned, but with low LR since
                  it's already pretrained on 15M PubMed abstracts)

Two learning rates used:
  Projection layer : higher LR (1e-4) — learning from scratch
  BioGPT layers     : lower LR (1e-5) — gentle fine-tuning,
                      avoid catastrophic forgetting of medical
                      language knowledge already learned

Run:
  python3 src/report_gen/train.py
"""

import os
import sys
import argparse
import torch
from torch.optim import AdamW
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.report_gen.model import MedReportGenerator
from src.data.dataset import XRayReportDataset, get_dataloader


def get_config():
    parser = argparse.ArgumentParser(description="Fine-tune BioGPT report generator")
    parser.add_argument("--vae_checkpoint", type=str, default="checkpoints/vae/vae_best.pth")
    parser.add_argument("--num_tokens",     type=int,   default=16)
    parser.add_argument("--batch_size",     type=int,   default=4)   # BioGPT is heavier than VAE/DDPM
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--lr_projection",  type=float, default=1e-4)
    parser.add_argument("--lr_biogpt",      type=float, default=1e-5)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--use_balanced",   type=str,   default="false")  # text matters → real reports only (see Phase 2 fix)
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints/report_gen")
    parser.add_argument("--patience",       type=int,   default=5)
    parser.add_argument("--gpu",            type=int,   default=1)
    args = parser.parse_args()
    args.use_balanced = args.use_balanced.lower() == "true"
    return args


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    # Keep VAE frozen and in eval mode even during model.train()
    model.vae.eval()

    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        image     = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(image, input_ids, labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0

    for batch in tqdm(loader, desc="Validating", leave=False):
        image     = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        outputs = model(image, input_ids, labels)
        total_loss += outputs.loss.item()

    return total_loss / len(loader)


def main():
    config = get_config()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*55)
    print("BioGPT Report Generator Training")
    print("="*55)
    print(f"Device       : {device}")
    print(f"Use balanced : {config.use_balanced}")
    print(f"Epochs       : {config.epochs}")
    print(f"Batch size   : {config.batch_size}\n")

    print("Loading datasets...")
    train_loader = get_dataloader(
        XRayReportDataset,
        split        = "train",
        batch_size   = config.batch_size,
        num_workers  = config.num_workers,
        use_balanced = config.use_balanced,
    )
    val_loader = get_dataloader(
        XRayReportDataset,
        split        = "val",
        batch_size   = config.batch_size,
        num_workers  = config.num_workers,
        use_balanced = False,
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}\n")

    print("Building model...")
    model = MedReportGenerator(
        vae_checkpoint = config.vae_checkpoint,
        num_tokens     = config.num_tokens,
        freeze_vae     = True,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters : {trainable:,} ({trainable/1e6:.2f}M)")
    print(f"Total parameters     : {total:,} ({total/1e6:.2f}M)\n")

    # Two parameter groups with different learning rates
    # Projection is new and needs to learn faster than BioGPT's
    # already-pretrained weights, which only need gentle nudging
    optimizer = AdamW([
        {"params": model.projection.parameters(), "lr": config.lr_projection},
        {"params": model.biogpt.parameters(),      "lr": config.lr_biogpt},
    ], weight_decay=1e-4)

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "biogpt_best.pth"

    best_loss        = float("inf")
    patience_counter = 0

    print("Starting training...\n")

    for epoch in range(config.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = validate(model, val_loader, device)

        print(f"Epoch {epoch+1:3d}/{config.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_loss - 1e-4:
            best_loss        = val_loss
            patience_counter  = 0
            torch.save({
                "epoch"               : epoch,
                "projection_state_dict": model.projection.state_dict(),
                "biogpt_state_dict"    : model.biogpt.state_dict(),
                "loss"                 : val_loss,
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
    print(f"Next: python3 src/report_gen/generate.py")


if __name__ == "__main__":
    main()