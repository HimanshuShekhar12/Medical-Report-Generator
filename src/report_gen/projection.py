"""
projection.py
--------------
Projects VAE latent vectors into BioGPT's embedding space.

Why needed?
  VAE latent z:        [B, 256, 16, 16]  (256 channels, spatial map)
  BioGPT expects:      [B, seq_len, 768] (sequence of 768-dim tokens)

This module bridges the two — converts the spatial latent map into
a short sequence of "visual tokens" that BioGPT can attend to,
exactly like how ClipCap/visual-prefix methods condition GPT-style
models on image features.

Pipeline:
  z [B, 256, 16, 16]
    → flatten spatial dims → [B, 256, 256]      (256 spatial positions)
    → pool down to N visual tokens → [B, 16, 256]
    → linear projection → [B, 16, 768]           (BioGPT hidden size)

These 16 visual tokens get prepended to the text token embeddings
before being fed into BioGPT — the model literally "reads" the
image as if it were the first 16 words of the report.
"""

import torch
import torch.nn as nn


class VisualProjection(nn.Module):
    """
    Maps VAE latent [B, 256, 16, 16] to BioGPT visual prefix [B, num_tokens, 768].

    Args:
        latent_channels : VAE latent depth (256)
        num_tokens       : how many visual tokens to produce (16)
        biogpt_dim       : BioGPT hidden size (768)
    """

    def __init__(
        self,
        latent_channels : int = 256,
        num_tokens       : int = 32,
        biogpt_dim       : int = 1024,
    ):
        super().__init__()
        self.num_tokens = num_tokens

        # Pool all spatial positions (16×16=256) down to exactly num_tokens.
        # 1D adaptive pool works for any num_tokens value — no perfect-square
        # requirement unlike the old 2D grid approach (which silently gave
        # int(sqrt(32))^2 = 25 tokens instead of 32).
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)

        self.projection = nn.Sequential(
            nn.Linear(latent_channels, biogpt_dim),
            nn.LayerNorm(biogpt_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(biogpt_dim, biogpt_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Input:  z [B, 256, 16, 16]   VAE latent
        Output:   [B, num_tokens, biogpt_dim]  visual tokens for BioGPT
        """
        B, C, H, W = z.shape
        z = z.flatten(2)        # [B, 256, 256]  flatten spatial dims
        z = self.pool(z)        # [B, 256, num_tokens]
        z = z.transpose(1, 2)  # [B, num_tokens, 256]
        return self.projection(z)


# ── Quick Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing VisualProjection...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proj   = VisualProjection(latent_channels=256, num_tokens=16, biogpt_dim=768).to(device)

    params = sum(p.numel() for p in proj.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")

    # Fake VAE latent
    z = torch.randn(2, 256, 16, 16).to(device)
    visual_tokens = proj(z)

    print(f"Input  (VAE latent)  : {z.shape}")
    print(f"Output (visual tokens): {visual_tokens.shape}")   # [2, 16, 768]

    assert visual_tokens.shape == (2, 16, 768)
    print("[OK] VisualProjection working!")
    print("\nNext: src/report_gen/model.py")