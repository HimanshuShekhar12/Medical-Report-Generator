# Decoder for VAE
import torch
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(
            self,
            latent_channels = 256,
            out_channels =1
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            # Block 1: [B, 256, 16, 16] → [B, 128, 32, 32]
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),

            # Block 2: [B, 128, 32, 32] → [B, 64, 64, 64]
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            # Block 3: [B, 64, 64, 64] → [B, 32, 128, 128]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),

            # Block 4: [B, 32, 128, 128] → [B, out_channels (e.g.,1), 256, 256]
            nn.ConvTranspose2d(32,out_channels , kernel_size=3,stride=2,padding=1 ,output_padding=1),
            nn.Sigmoid()  # Output pixel values in [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
            # Decode latent representation to image
            x_recon = self.decoder(z)  # [B, out_channels, 256, 256]
            return x_recon

#Quick test
if __name__ == "__main__":
    print("Testing Decoder...")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = Decoder(latent_channels=256, out_channels=1).to(device)

    # Fake latent vector (what encoder outputs)
    z = torch.randn(2, 256, 16, 16).to(device)

    # Forward pass
    out = decoder.forward(z)

    print(f"Input shape  : {z.shape}")
    print(f"Output shape : {out.shape}")    # should be [2, 1, 256, 256]
    print(f"Output range : [{out.min():.3f}, {out.max():.3f}]")  # should be [0, 1]

    assert out.shape == (2, 1, 256, 256)
    print("[OK] Decoder working!")
    