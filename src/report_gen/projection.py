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
        num_tokens       : int = 16,
        biogpt_dim       : int = 768,
    ):
        super().__init__()
        self.num_tokens = num_tokens

        # Step 1: pool 16x16 spatial map down to `num_tokens` positions
        # AdaptiveAvgPool2d can target any output size, here we pick
        # a 4x4 grid = 16 tokens (matches num_tokens default)
        grid_size = int(num_tokens ** 0.5)   # 16 → 4x4 grid
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        # Step 2: project channel depth (256) to BioGPT hidden size (768)
        self.projection = nn.Sequential(
            nn.Linear(latent_channels, biogpt_dim),
            nn.LayerNorm(biogpt_dim),
            nn.GELU(),
            nn.Linear(biogpt_dim, biogpt_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Input:  z [B, 256, 16, 16]   VAE latent
        Output:   [B, num_tokens, 768]  visual tokens for BioGPT
        """
        B, C, H, W = z.shape

        # Step 1: pool spatial dims down to grid_size x grid_size
        z = self.pool(z)                          # [B, 256, 4, 4]

        # Step 2: flatten spatial grid into a sequence
        z = z.flatten(2)                           # [B, 256, 16]
        z = z.transpose(1, 2)                      # [B, 16, 256]  (seq_len, channels)

        # Step 3: project each token's channel vector to BioGPT dim
        visual_tokens = self.projection(z)         # [B, 16, 768]

        return visual_tokens


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