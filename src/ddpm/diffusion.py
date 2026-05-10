"""
diffusion.py
------------
Forward and reverse diffusion process for DDPM.

FORWARD PROCESS (training):
  Takes a clean X-ray → adds noise step by step → noisy image
  q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

REVERSE PROCESS (inference/sampling):
  Takes pure noise → removes noise step by step → clean X-ray
  p(x_{t-1} | x_t) = UNet predicts noise → subtract it

Key variables:
  T          = total timesteps (1000)
  beta_t     = noise schedule (how much noise at each step)
  alpha_t    = 1 - beta_t
  alpha_bar_t = product of all alphas up to t (cumulative)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class DDPM(nn.Module):
    """
    Full DDPM model combining:
      - Noise scheduler (forward process)
      - UNet (reverse process / noise predictor)
      - Conditioning (timestep + class embeddings)

    Usage:
      ddpm = DDPM(unet, conditioning)

      # Training:
      loss = ddpm.training_loss(clean_images, class_labels)

      # Sampling:
      samples = ddpm.sample(class_labels, num_samples=10)
    """

    def __init__(
        self,
        unet,
        conditioning,
        timesteps    : int   = 1000,
        beta_start   : float = 1e-4,   # noise at t=1 (very small)
        beta_end     : float = 0.02,   # noise at t=T (larger)
        device       : str   = "cuda",
    ):
        super().__init__()

        self.unet         = unet
        self.conditioning = conditioning
        self.timesteps    = timesteps
        self.device       = device

        # -- Build noise schedule --
        # Linear schedule: beta increases linearly from beta_start to beta_end
        # This is the original DDPM schedule
        betas = self._linear_beta_schedule(timesteps, beta_start, beta_end)

        # Precompute all the constants we need
        # (done once at init, not every training step — much faster)
        alphas          = 1.0 - betas
        alphas_cumprod  = np.cumprod(alphas)                    # alpha_bar_t
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])  # alpha_bar_{t-1}

        # Register as buffers (moved to GPU with model, but not trained)
        self.register_buffer("betas",       torch.FloatTensor(betas))
        self.register_buffer("alphas",      torch.FloatTensor(alphas))
        self.register_buffer("alphas_cumprod",     torch.FloatTensor(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev",torch.FloatTensor(alphas_cumprod_prev))

        # Precomputed terms for forward process q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod",
            torch.FloatTensor(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
            torch.FloatTensor(np.sqrt(1.0 - alphas_cumprod)))

        # Precomputed terms for reverse process p(x_{t-1} | x_t)
        self.register_buffer("sqrt_recip_alphas",
            torch.FloatTensor(np.sqrt(1.0 / alphas)))
        self.register_buffer("posterior_variance",
            torch.FloatTensor(betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)))

    # ---------------------------------------------------------------- #
    #  NOISE SCHEDULE                                                   #
    # ---------------------------------------------------------------- #
    @staticmethod
    def _linear_beta_schedule(
        timesteps  : int,
        beta_start : float,
        beta_end   : float,
    ) -> np.ndarray:
        """
        Linear schedule: beta increases from beta_start to beta_end.

        t=1:    beta = 0.0001  (add tiny noise)
        t=500:  beta = 0.010   (add medium noise)
        t=1000: beta = 0.02    (add lots of noise)

        After t=1000: image is essentially pure Gaussian noise.
        """
        return np.linspace(beta_start, beta_end, timesteps)

    # ---------------------------------------------------------------- #
    #  FORWARD PROCESS: add noise to clean image                       #
    # ---------------------------------------------------------------- #
    def add_noise(
        self,
        x_0 : torch.Tensor,   # clean image [B, 1, H, W]
        t   : torch.Tensor,   # timestep [B]
        noise: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Forward process: x_0 → x_t (add noise)

        Uses the closed-form formula (no need to loop through all timesteps):
          x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        This is the KEY mathematical insight of DDPM:
        You can directly compute the noisy image at ANY timestep t
        without going through t-1, t-2, ... 1 one by one.

        Returns:
          x_t   : noisy image at timestep t
          noise : the actual noise that was added (UNet tries to predict this)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Extract the right constants for each sample's timestep
        # t is [B], we need to index into our precomputed arrays
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # Forward process formula
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus * noise

        return x_t, noise

    # ---------------------------------------------------------------- #
    #  TRAINING LOSS                                                    #
    # ---------------------------------------------------------------- #
    def training_loss(
        self,
        x_0       : torch.Tensor,   # clean images [B, 1, H, W]
        class_idx : torch.Tensor,   # class labels [B]
    ) -> torch.Tensor:
        """
        Computes DDPM training loss.

        Steps:
          1. Sample random timesteps t for each image in batch
          2. Add noise to get x_t
          3. Get conditioning vector (timestep + class)
          4. UNet predicts the noise
          5. Loss = MSE(predicted_noise, actual_noise)

        Why MSE on noise?
          Original DDPM paper showed predicting noise (epsilon)
          works better than predicting x_0 directly.
          The UNet learns: "given this noisy image at this timestep
          for this disease class, what noise was added?"
        """
        B = x_0.shape[0]

        # Step 1: Sample random timesteps (different for each image in batch)
        t = torch.randint(0, self.timesteps, (B,), device=x_0.device)

        # Step 2: Add noise → get noisy image + actual noise
        x_t, noise = self.add_noise(x_0, t)

        # Step 3: Get conditioning vector
        cond = self.conditioning(t, class_idx)  # [B, 256]

        # Step 4: UNet predicts the noise
        noise_pred = self.unet(x_t, cond)       # [B, 1, H, W]

        # Step 5: MSE loss between predicted and actual noise
        loss = nn.functional.mse_loss(noise_pred, noise)

        return loss

    # ---------------------------------------------------------------- #
    #  REVERSE PROCESS: denoise step by step (sampling)                #
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def sample(
        self,
        class_idx     : torch.Tensor,   # [B] which classes to generate
        image_size    : int = 256,
        guidance_scale: float = 3.0,    # classifier-free guidance strength
    ) -> torch.Tensor:
        """
        Reverse process: pure noise → clean X-ray.

        Loops from t=T down to t=1, each step:
          1. UNet predicts noise in x_t
          2. Subtract predicted noise to get x_{t-1}
          3. Add small random noise (except at t=1)

        Classifier-free guidance:
          Run UNet twice per step:
            - conditional:   with class label
            - unconditional: with null class label (num_classes index)
          Final prediction = uncond + scale * (cond - uncond)
          Higher scale = stronger class guidance (more disease-specific)

        Returns: generated X-rays [B, 1, H, W] in range [-1, 1]
        """
        B      = class_idx.shape[0]
        device = class_idx.device

        # Start from pure Gaussian noise
        x = torch.randn(B, 1, image_size, image_size, device=device)

        # Unconditional class index (for classifier-free guidance)
        # num_classes index = "null" class in ClassEmbedding
        num_classes = self.conditioning.class_embed.embedding.num_embeddings - 1
        uncond_idx  = torch.full((B,), num_classes, device=device, dtype=torch.long)

        # Reverse loop: T → 1
        for t_val in reversed(range(self.timesteps)):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # -- Conditional prediction --
            cond       = self.conditioning(t, class_idx)
            noise_cond = self.unet(x, cond)

            # -- Unconditional prediction (classifier-free guidance) --
            uncond      = self.conditioning(t, uncond_idx)
            noise_uncond = self.unet(x, uncond)

            # -- Classifier-free guidance --
            # Interpolate between unconditional and conditional
            # Higher guidance_scale = more class-specific output
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            # -- Compute x_{t-1} from x_t and predicted noise --
            x = self._reverse_step(x, t, noise_pred)

        # Clamp to [-1, 1]
        return x.clamp(-1, 1)

    def _reverse_step(
        self,
        x_t        : torch.Tensor,   # current noisy image [B, 1, H, W]
        t          : torch.Tensor,   # current timestep [B]
        noise_pred : torch.Tensor,   # UNet's noise prediction [B, 1, H, W]
    ) -> torch.Tensor:
        """
        One step of reverse process:
          x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * noise_pred)
                    + sqrt(posterior_variance_t) * z

        where z ~ N(0, I) is random noise (added except at t=0).
        """
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas,   t, x_t.shape)
        beta_t           = self._extract(self.betas,               t, x_t.shape)
        sqrt_one_minus   = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        # Mean of x_{t-1}
        mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_one_minus * noise_pred)

        # Add noise (except at final step t=0)
        if t[0].item() > 0:
            posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
            z    = torch.randn_like(x_t)
            mean = mean + torch.sqrt(posterior_var) * z

        return mean

    # ---------------------------------------------------------------- #
    #  UTILITY                                                          #
    # ---------------------------------------------------------------- #
    @staticmethod
    def _extract(
        arr    : torch.Tensor,
        t      : torch.Tensor,
        shape  : tuple,
    ) -> torch.Tensor:
        """
        Extracts values from arr at indices t, then reshapes for broadcasting.

        arr: [T] — precomputed constants for all timesteps
        t:   [B] — which timestep each sample is at
        Returns: [B, 1, 1, 1] for broadcasting with [B, C, H, W]
        """
        B    = t.shape[0]
        vals = arr[t]
        # Reshape: [B] → [B, 1, 1, 1] for broadcasting with images
        return vals.reshape(B, *((1,) * (len(shape) - 1)))


# ------------------------------------------------------------------ #
#  QUICK TEST                                                        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    from unet import UNet
    from conditioning import Conditioning

    print("Testing DDPM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    unet        = UNet(in_channels=1, base_channels=16, cond_dim=128).to(device)
    conditioning = Conditioning(num_classes=14, dim=128).to(device)
    ddpm        = DDPM(unet, conditioning, timesteps=100, device=str(device)).to(device)

    # Test forward process
    x_0       = torch.randn(2, 1, 64, 64).to(device)   # fake clean images
    class_idx = torch.tensor([0, 4]).to(device)          # normal, fibrosis

    # Test noise addition
    t     = torch.tensor([50, 99]).to(device)
    x_t, noise = ddpm.add_noise(x_0, t)
    print(f"Clean image range : [{x_0.min():.2f}, {x_0.max():.2f}]")
    print(f"Noisy image range : [{x_t.min():.2f}, {x_t.max():.2f}]")

    # Test training loss
    loss = ddpm.training_loss(x_0, class_idx)
    print(f"Training loss: {loss.item():.4f}")

    # Test sampling (small scale)
    samples = ddpm.sample(class_idx, image_size=64)
    print(f"Generated samples shape: {samples.shape}")

    print("[OK] DDPM working!")