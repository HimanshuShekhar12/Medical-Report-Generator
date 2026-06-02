import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels     : int = 1,
        latent_channels : int = 256,
    ):
        super().__init__()
        self.encoder = nn.Sequential(

            # Block 1: [B, 1, 256, 256] → [B, 32, 128, 128]
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),     # 32 channels / 8 groups = 4 channels per group
            nn.SiLU(),

            # Block 2: [B, 32, 128, 128] → [B, 64, 64, 64]
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            # Block 3: [B, 64, 64, 64] → [B, 128, 32, 32]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),

            # Block 4: [B, 128, 32, 32] → [B, 256, 16, 16]
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, latent_channels),
            nn.SiLU(),
        )

        # ── Two Separate Heads for mu and logvar ──────────────────────
        # kernel_size=1 = pointwise conv = acts like Linear per spatial location
        # No spatial downsampling — just channel projection
        # Both output [B, latent_channels, 16, 16]
        self.mu_head     = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.logvar_head = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)

    def reparameterize(
        self,
        mu     : torch.Tensor,   # [B, C, H, W] mean
        logvar : torch.Tensor,   # [B, C, H, W] log variance
    ) -> torch.Tensor:
        
        std = torch.exp(0.5 * logvar)   # convert log variance → std deviation
        eps = torch.randn_like(std)      # sample noise from N(0,1), same shape as std
        z   = mu + std * eps             # reparameterized sample
        return z

    def forward(self, x: torch.Tensor) -> tuple:
        # Step 1: Encode image to feature map
        h = self.encoder(x)           # [B, 256, 16, 16]

        # Step 2: Get distribution parameters
        mu     = self.mu_head(h)      # [B, 256, 16, 16]
        logvar = self.logvar_head(h)  # [B, 256, 16, 16]

        # Clamp logvar for numerical stability
        # Prevents exp(logvar) from exploding or vanishing
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)

        # Step 3: Sample z using reparameterization trick
        z = self.reparameterize(mu, logvar)   # [B, 256, 16, 16]

        return z, mu, logvar


# ── Quick Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Encoder...")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(in_channels=1, latent_channels=256).to(device)

    # Count parameters
    params = sum(p.numel() for p in encoder.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")

    # Fake batch
    x = torch.randn(2, 1, 256, 256).to(device)

    # Forward pass
    z, mu, logvar = encoder(x)

    print(f"Input shape  : {x.shape}")
    print(f"z shape      : {z.shape}")       # [2, 256, 16, 16]
    print(f"mu shape     : {mu.shape}")      # [2, 256, 16, 16]
    print(f"logvar shape : {logvar.shape}")  # [2, 256, 16, 16]

    assert z.shape      == (2, 256, 16, 16)
    assert mu.shape     == (2, 256, 16, 16)
    assert logvar.shape == (2, 256, 16, 16)

    print("[OK] Encoder working!")
