"""
unet.py
-------
U-Net architecture for DDPM noise prediction.

U-Net is the neural network INSIDE the diffusion model.
Its job: given a noisy image + timestep + class → predict the noise.

Architecture:
                    [noisy image]
                         ↓
                    [Encoder]
              ┌──────────────────────┐
              │  ResBlock + Attention │  ← skip connection saved
              │  Downsample          │
              │  ResBlock + Attention │  ← skip connection saved
              │  Downsample          │
              └──────────────────────┘
                         ↓
                    [Bottleneck]
              ┌──────────────────────┐
              │  ResBlock            │
              │  Attention           │
              │  ResBlock            │
              └──────────────────────┘
                         ↓
                    [Decoder]
              ┌──────────────────────┐
              │  ResBlock + Attention │  ← skip connection added back
              │  Upsample            │
              │  ResBlock + Attention │  ← skip connection added back
              │  Upsample            │
              └──────────────────────┘
                         ↓
                  [predicted noise]

The conditioning vector (timestep + class) is injected into
every ResBlock via a learned scale+shift (AdaGN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  BASIC BUILDING BLOCKS                                             #
# ------------------------------------------------------------------ #

class GroupNorm(nn.Module):
    """
    Group Normalization — standard for diffusion models.
    Better than BatchNorm for small batches (common in high-res training).
    """
    def __init__(self, channels: int):
        super().__init__()
        # 32 groups is standard for diffusion UNets
        num_groups = min(32, channels)
        # Make sure channels is divisible by num_groups
        while channels % num_groups != 0:
            num_groups //= 2
        self.norm = nn.GroupNorm(num_groups, channels)

    def forward(self, x):
        return self.norm(x)


class ResBlock(nn.Module):
    """
    Residual Block with conditioning injection.

    This is the core building block of the UNet.
    Takes image features + conditioning vector → refined features.

    Conditioning is injected via scale+shift (AdaGN):
      norm(x) * (1 + scale) + shift
      where scale and shift come from the conditioning vector.

    This is how the UNet "knows" what timestep and disease class it's at.
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int = 256):
        super().__init__()

        # First conv block
        self.norm1 = GroupNorm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Conditioning projection → scale and shift for AdaGN
        # cond_dim → out_channels * 2 (half for scale, half for shift)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)

        # Second conv block
        self.norm2 = GroupNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.act = nn.SiLU()

        # Skip connection: match channels if in != out
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x    : [B, in_channels, H, W]  — image features
        cond : [B, cond_dim]           — conditioning vector
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Inject conditioning via scale + shift
        # cond_proj: [B, cond_dim] → [B, out_channels * 2]
        cond_out = self.cond_proj(self.act(cond))
        # Split into scale and shift: each [B, out_channels]
        scale, shift = cond_out.chunk(2, dim=1)
        # Reshape for broadcasting: [B, C] → [B, C, 1, 1]
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        # Apply AdaGN: norm(h) * (1 + scale) + shift
        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)

        # Add skip connection
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing long-range dependencies.

    In X-ray images, findings in one part of the lung affect
    how we interpret findings in another part.
    Attention lets the model capture these spatial relationships.

    Only used at lower resolutions (32×32, 16×16) for efficiency.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = GroupNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim   = channels,
            num_heads   = max(1, channels // 32),  # 1 head per 32 channels
            batch_first = True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Reshape: [B, C, H, W] → [B, H*W, C]
        # Treat each spatial location as a "token" (like Transformer)
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]

        # Self-attention
        h, _ = self.attn(h, h, h)

        # Reshape back: [B, H*W, C] → [B, C, H, W]
        h = h.permute(0, 2, 1).reshape(B, C, H, W)

        return x + h   # residual connection


class Downsample(nn.Module):
    """Halves spatial dimensions: [B, C, H, W] → [B, C, H/2, W/2]"""
    def __init__(self, channels: int):
        super().__init__()
        # Strided conv (better than MaxPool for learned downsampling)
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Doubles spatial dimensions: [B, C, H, W] → [B, C, H*2, W*2]"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        # Nearest neighbor upsample then conv (avoids checkerboard artifacts)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ------------------------------------------------------------------ #
#  FULL UNET                                                         #
# ------------------------------------------------------------------ #
class UNet(nn.Module):
    """
    Full U-Net for DDPM noise prediction.

    Input:
      x         : [B, 1, 256, 256]  noisy X-ray (1 channel = grayscale)
      cond      : [B, cond_dim]     conditioning (timestep + class)

    Output:
      predicted noise: [B, 1, 256, 256]  same shape as input

    Channel progression (encoder):
      256×256 → 128×128 → 64×64 → 32×32 → 16×16

    Channels at each level:
      [32, 64, 128, 256, 512]
    """

    def __init__(
        self,
        in_channels : int = 1,      # grayscale X-ray
        base_channels: int = 32,    # start small, double at each level
        cond_dim    : int = 256,    # conditioning vector size
        num_classes : int = 14,     # number of disease classes
    ):
        super().__init__()

        # Channel sizes at each resolution level
        ch = [base_channels, base_channels*2, base_channels*4,
              base_channels*8, base_channels*16]
        # ch = [32, 64, 128, 256, 512]

        # -- Initial projection --
        # Project 1-channel image to base_channels feature map
        self.init_conv = nn.Conv2d(in_channels, ch[0], 3, padding=1)

        # -- Encoder (downsampling path) --
        self.enc1 = nn.ModuleList([
            ResBlock(ch[0], ch[0], cond_dim),
            ResBlock(ch[0], ch[0], cond_dim),
        ])
        self.down1 = Downsample(ch[0])   # 256 → 128

        self.enc2 = nn.ModuleList([
            ResBlock(ch[0], ch[1], cond_dim),
            ResBlock(ch[1], ch[1], cond_dim),
        ])
        self.down2 = Downsample(ch[1])   # 128 → 64

        self.enc3 = nn.ModuleList([
            ResBlock(ch[1], ch[2], cond_dim),
            ResBlock(ch[2], ch[2], cond_dim),
            AttentionBlock(ch[2]),        # attention at 64×64
        ])
        self.down3 = Downsample(ch[2])   # 64 → 32

        self.enc4 = nn.ModuleList([
            ResBlock(ch[2], ch[3], cond_dim),
            ResBlock(ch[3], ch[3], cond_dim),
            AttentionBlock(ch[3]),        # attention at 32×32
        ])
        self.down4 = Downsample(ch[3])   # 32 → 16

        # -- Bottleneck --
        self.bottleneck = nn.ModuleList([
            ResBlock(ch[3], ch[4], cond_dim),
            AttentionBlock(ch[4]),        # attention at 16×16
            ResBlock(ch[4], ch[4], cond_dim),
        ])

        # -- Decoder (upsampling path) --
        # Note: decoder input = upsampled + skip connection (concatenated)
        # So first ResBlock has doubled input channels
        self.up4 = Upsample(ch[4])
        self.dec4 = nn.ModuleList([
            ResBlock(ch[4] + ch[3], ch[3], cond_dim),  # +skip
            ResBlock(ch[3], ch[3], cond_dim),
            AttentionBlock(ch[3]),
        ])

        self.up3 = Upsample(ch[3])
        self.dec3 = nn.ModuleList([
            ResBlock(ch[3] + ch[2], ch[2], cond_dim),  # +skip
            ResBlock(ch[2], ch[2], cond_dim),
            AttentionBlock(ch[2]),
        ])

        self.up2 = Upsample(ch[2])
        self.dec2 = nn.ModuleList([
            ResBlock(ch[2] + ch[1], ch[1], cond_dim),  # +skip
            ResBlock(ch[1], ch[1], cond_dim),
        ])

        self.up1 = Upsample(ch[1])
        self.dec1 = nn.ModuleList([
            ResBlock(ch[1] + ch[0], ch[0], cond_dim),  # +skip
            ResBlock(ch[0], ch[0], cond_dim),
        ])

        # -- Output projection --
        self.out_norm = GroupNorm(ch[0])
        self.out_conv = nn.Conv2d(ch[0], in_channels, 1)  # back to 1 channel
        self.act = nn.SiLU()

    def forward(
        self,
        x    : torch.Tensor,   # [B, 1, 256, 256] noisy image
        cond : torch.Tensor,   # [B, 256] conditioning vector
    ) -> torch.Tensor:
        """
        Returns predicted noise: [B, 1, 256, 256]
        Same shape as input — UNet predicts what noise was added.
        """

        # -- Initial conv --
        x = self.init_conv(x)   # [B, 32, 256, 256]

        # -- Encoder --
        # Save skip connections at each resolution
        h1 = x
        for block in self.enc1:
            h1 = block(h1, cond) if isinstance(block, ResBlock) else block(h1)
        skip1 = h1                          # [B, 32, 256, 256]
        h1 = self.down1(h1)                 # [B, 32, 128, 128]

        h2 = h1
        for block in self.enc2:
            h2 = block(h2, cond) if isinstance(block, ResBlock) else block(h2)
        skip2 = h2                          # [B, 64, 128, 128]
        h2 = self.down2(h2)                 # [B, 64, 64, 64]

        h3 = h2
        for block in self.enc3:
            h3 = block(h3, cond) if isinstance(block, ResBlock) else block(h3)
        skip3 = h3                          # [B, 128, 64, 64]
        h3 = self.down3(h3)                 # [B, 128, 32, 32]

        h4 = h3
        for block in self.enc4:
            h4 = block(h4, cond) if isinstance(block, ResBlock) else block(h4)
        skip4 = h4                          # [B, 256, 32, 32]
        h4 = self.down4(h4)                 # [B, 256, 16, 16]

        # -- Bottleneck --
        h = h4
        for block in self.bottleneck:
            h = block(h, cond) if isinstance(block, ResBlock) else block(h)
                                            # [B, 512, 16, 16]

        # -- Decoder --
        # Each level: upsample → concat skip → ResBlocks
        h = self.up4(h)                                 # [B, 512, 32, 32]
        h = torch.cat([h, skip4], dim=1)                # [B, 768, 32, 32]
        for block in self.dec4:
            h = block(h, cond) if isinstance(block, ResBlock) else block(h)
                                                        # [B, 256, 32, 32]

        h = self.up3(h)                                 # [B, 256, 64, 64]
        h = torch.cat([h, skip3], dim=1)                # [B, 384, 64, 64]
        for block in self.dec3:
            h = block(h, cond) if isinstance(block, ResBlock) else block(h)
                                                        # [B, 128, 64, 64]

        h = self.up2(h)                                 # [B, 128, 128, 128]
        h = torch.cat([h, skip2], dim=1)                # [B, 192, 128, 128]
        for block in self.dec2:
            h = block(h, cond) if isinstance(block, ResBlock) else block(h)
                                                        # [B, 64, 128, 128]

        h = self.up1(h)                                 # [B, 64, 256, 256]
        h = torch.cat([h, skip1], dim=1)                # [B, 96, 256, 256]
        for block in self.dec1:
            h = block(h, cond) if isinstance(block, ResBlock) else block(h)
                                                        # [B, 32, 256, 256]

        # -- Output --
        h = self.act(self.out_norm(h))
        h = self.out_conv(h)                            # [B, 1, 256, 256]
        return h


# ------------------------------------------------------------------ #
#  QUICK TEST                                                        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("Testing UNet...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UNet(
        in_channels  = 1,
        base_channels = 32,
        cond_dim     = 256,
        num_classes  = 14,
    ).to(device)

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")

    # Fake batch
    x    = torch.randn(2, 1, 256, 256).to(device)  # noisy images
    cond = torch.randn(2, 256).to(device)           # conditioning vectors

    # Forward pass
    with torch.no_grad():
        noise_pred = model(x, cond)

    print(f"Input shape  : {x.shape}")
    print(f"Output shape : {noise_pred.shape}")  # should match input
    assert x.shape == noise_pred.shape, "Shape mismatch!"
    print("[OK] UNet working!")