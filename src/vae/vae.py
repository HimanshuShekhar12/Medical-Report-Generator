# Variational Autoencoder Implementation(Loss function and KL Divergence)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.vae.decoder import Decoder
from src.vae.encoder import Encoder
import torch 
import torch.nn as nn

class VAE(nn.Module):
    
    def __init__(
        self,
        in_channels     : int = 1,
        latent_channels : int = 256,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels)
        self.decoder = Decoder(latent_channels, in_channels)

    def forward(self, x: torch.Tensor) -> tuple:
        # Encode input image to get mu and logvar
        z,mu, logvar = self.encoder(x)  # [B, 256, 16, 16] each


        # Decode z to reconstruct the image
        x_recon = self.decoder(z)  # [B, 1, 256, 256]

        return x_recon, mu, logvar
    
    def loss_function(self, x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')

        # KL Divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total VAE loss
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss
    
if __name__ == "__main__":
    print("Testing VAE...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae    = VAE(in_channels=1, latent_channels=256).to(device)

    # Count parameters
    params = sum(p.numel() for p in vae.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")

    # Fake batch
    x = torch.randn(2, 1, 256, 256).to(device)

    # Forward pass
    x_recon, mu, logvar = vae(x)

    print(f"Input shape  : {x.shape}")
    print(f"Recon shape  : {x_recon.shape}")
    print(f"mu shape     : {mu.shape}")

    # Loss
    total, recon, kl = vae.loss_function(x_recon, x, mu, logvar)
    print(f"Recon loss   : {recon.item():.4f}")
    print(f"KL loss      : {kl.item():.4f}")
    print(f"Total loss   : {total.item():.4f}")
    print("[OK] VAE working!")
