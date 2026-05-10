"""
conditioning.py
---------------
Timestep and class embeddings for conditional DDPM.

Two embeddings are created and ADDED together:
  1. Timestep embedding  → tells UNet which noise level it's at
  2. Class embedding     → tells UNet which disease class to generate

Both are 256-dimensional vectors injected into every UNet block.

Key insight:
  Timestep embedding = sinusoidal (same as Transformer PE, fixed formula)
  Class embedding    = nn.Embedding (learnable, trained with DDPM)
"""

import math
import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
#  TIMESTEP EMBEDDING (Sinusoidal — same as Transformer PE)          #
# ------------------------------------------------------------------ #
class TimestepEmbedding(nn.Module):
    """
    Converts integer timestep t → 256-dim vector.

    Uses sinusoidal encoding (identical to Transformer positional encoding).
    Then passes through 2 linear layers to make it learnable.

    Why sinusoidal?
      - Works well for ordered sequences (t=1,2,...,1000)
      - Each timestep gets a unique vector
      - Nearby timesteps get similar vectors (t=500 similar to t=501)
      - Same reason it works in Transformers for word positions
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim

        # Two linear layers refine the sinusoidal embedding
        # Makes it learnable on top of the fixed sinusoidal base
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.act     = nn.SiLU()   # SiLU (Swish) works better than ReLU for diffusion

    def sinusoidal_encode(self, t: torch.Tensor) -> torch.Tensor:
        """
        Converts integer timesteps → sinusoidal vectors.

        Same formula as Transformer PE:
          PE(t, 2i)   = sin(t / 10000^(2i/dim))
          PE(t, 2i+1) = cos(t / 10000^(2i/dim))

        Input:  t shape [batch_size]  (integer timesteps)
        Output: shape [batch_size, dim]
        """
        half = self.dim // 2

        # Frequency terms (same as Transformer PE)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )

        # Outer product: [batch, 1] * [1, half] → [batch, half]
        args = t[:, None].float() * freqs[None, :]

        # Concatenate sin and cos → [batch, dim]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Input:  t [batch_size] — integer timesteps e.g. tensor([500, 23, 891])
        Output:   [batch_size, dim] — embedding vectors
        """
        x = self.sinusoidal_encode(t)   # [B, dim] — fixed sinusoidal
        x = self.linear1(x)             # [B, dim*4] — learned refinement
        x = self.act(x)
        x = self.linear2(x)             # [B, dim] — final embedding
        return x


# ------------------------------------------------------------------ #
#  CLASS EMBEDDING (Learnable — like word embeddings)                #
# ------------------------------------------------------------------ #
class ClassEmbedding(nn.Module):
    """
    Converts disease class index → 256-dim vector.

    Exactly like nn.Embedding in NLP:
      class 0 (normal)    → [0.23, -0.41, 0.87, ...]
      class 1 (cardiomeg) → [0.11,  0.93, -0.2, ...]
      class 4 (fibrosis)  → [0.77, -0.22, 0.65, ...]

    These vectors are LEARNED during DDPM training.
    After training, similar diseases will have similar embeddings.

    Also supports classifier-free guidance:
      class_idx = num_classes → "unconditional" (no class guidance)
      Used during inference for guidance scale trick.
    """

    def __init__(self, num_classes: int = 14, dim: int = 256):
        super().__init__()

        # +1 for unconditional token (used in classifier-free guidance)
        self.embedding = nn.Embedding(num_classes + 1, dim)
        self.linear    = nn.Linear(dim, dim)
        self.act       = nn.SiLU()

    def forward(self, class_idx: torch.Tensor) -> torch.Tensor:
        """
        Input:  class_idx [batch_size] — integer class indices
                e.g. tensor([4, 0, 1]) → fibrosis, normal, cardiomegaly
        Output: [batch_size, dim] — class embedding vectors
        """
        x = self.embedding(class_idx)  # [B, dim]
        x = self.act(self.linear(x))   # [B, dim]
        return x


# ------------------------------------------------------------------ #
#  COMBINED CONDITIONING                                             #
# ------------------------------------------------------------------ #
class Conditioning(nn.Module):
    """
    Combines timestep + class embeddings into one conditioning vector.

    Final conditioning = timestep_embed + class_embed
    This single vector is injected into every UNet residual block.

    Usage:
        conditioning = Conditioning(num_classes=14, dim=256)
        cond = conditioning(t=timesteps, class_idx=labels)
        # cond shape: [batch, 256]
        # inject into UNet blocks
    """

    def __init__(self, num_classes: int = 14, dim: int = 256):
        super().__init__()
        self.timestep_embed = TimestepEmbedding(dim)
        self.class_embed    = ClassEmbedding(num_classes, dim)

    def forward(
        self,
        t         : torch.Tensor,   # [B] integer timesteps
        class_idx : torch.Tensor,   # [B] integer class indices
    ) -> torch.Tensor:
        """
        Returns combined conditioning vector [B, dim].
        UNet adds this to its hidden states at every block.
        """
        t_emb = self.timestep_embed(t)          # [B, dim]
        c_emb = self.class_embed(class_idx)     # [B, dim]
        return t_emb + c_emb                    # [B, dim]


# ------------------------------------------------------------------ #
#  QUICK TEST                                                        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("Testing Conditioning module...")

    conditioning = Conditioning(num_classes=14, dim=256)

    # Fake batch: 4 samples
    timesteps  = torch.tensor([100, 500, 999, 1])     # different noise levels
    class_idxs = torch.tensor([0, 4, 1, 13])          # normal, fibrosis, cardio, other

    cond = conditioning(timesteps, class_idxs)

    print(f"Timesteps  : {timesteps}")
    print(f"Classes    : {class_idxs}")
    print(f"Output shape: {cond.shape}")   # should be [4, 256]
    print(f"First 5 values of sample 0: {cond[0, :5]}")
    print("[OK] Conditioning working!")