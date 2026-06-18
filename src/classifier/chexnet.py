"""
chexnet.py
----------
DenseNet121-based multi-label pathology classifier.

Purpose in pipeline:
  1. Ablation study — prove DDPM synthetic data improved rare-class F1
  2. Consistency check gate — verify BioGPT report matches image findings
  3. Pathology conditioning — feed predicted labels into BioGPT

Architecture:
  Input:  [B, 1, 256, 256]  grayscale X-ray
  → repeat channel 3x → [B, 3, 256, 256]  (DenseNet expects RGB)
  → DenseNet121 backbone (ImageNet pretrained)
  → Global Average Pool → [B, 1024]
  → Dropout(0.4)
  → Linear(1024, 18)
  → Sigmoid → [B, 18] independent probabilities

Why DenseNet121?
  Dense connections: each layer receives feature maps from ALL
  previous layers (not just the one before it like ResNet).
  This improves gradient flow and feature reuse — well suited
  for medical imaging where subtle features matter.

Why Sigmoid not Softmax?
  Softmax forces probabilities to sum to 1 — implies mutually
  exclusive classes. But a patient can have BOTH cardiomegaly
  AND effusion simultaneously. Sigmoid treats each class as an
  independent binary decision — correct for multi-label setup.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CheXNet(nn.Module):
    """
    DenseNet121-based multi-label chest X-ray classifier.

    Args:
        num_classes : number of pathology classes (18 for our LABEL_MAP)
        pretrained  : use ImageNet pretrained weights
        dropout     : dropout rate before final linear layer
    """

    def __init__(
        self,
        num_classes : int   = 18,
        pretrained  : bool  = True,
        dropout     : float = 0.4,
    ):
        super().__init__()

        # Load DenseNet121 backbone
        self.densenet = models.densenet121(pretrained=pretrained)

        # DenseNet121's classifier expects 1024 input features
        num_features = self.densenet.classifier.in_features  # 1024

        # Replace the original 1000-class ImageNet head with our own
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x [B, 1, 256, 256]  grayscale X-ray, normalized [-1,1] or [0,1]
        Output: probs [B, num_classes]  independent pathology probabilities
        """
        # DenseNet expects 3-channel RGB input
        # Our X-rays are single-channel grayscale → repeat to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)   # [B, 1, H, W] → [B, 3, H, W]

        probs = self.densenet(x)   # [B, num_classes]
        return probs


# ── Quick Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing CheXNet...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CheXNet(num_classes=18).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")

    x = torch.randn(2, 1, 256, 256).to(device)
    out = model(x)

    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {out.shape}")          # [2, 18]
    print(f"Output range : [{out.min():.3f}, {out.max():.3f}]")  # [0, 1]

    assert out.shape == (2, 18)
    print("[OK] CheXNet working!")
    print("\nNext: python3 src/classifier/train.py")