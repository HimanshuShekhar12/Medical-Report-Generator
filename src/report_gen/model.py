"""
model.py
--------
Combines VAE visual tokens with BioGPT to generate radiology reports.

Architecture:
  X-ray → VAE encoder → z [B,256,16,16]
        → VisualProjection → visual_tokens [B,16,768]
        → prepended to BioGPT text embeddings
        → BioGPT generates report tokens autoregressively

This is the "visual prefix" approach (same idea as ClipCap):
  Instead of modifying BioGPT's internal attention mechanism,
  we simply prepend image information as if it were extra
  "words" at the start of the sequence. BioGPT's existing
  self-attention naturally lets every text token attend back
  to these visual tokens — no architecture surgery needed.

Why this approach over modifying BioGPT's attention?
  - Keeps BioGPT's pretrained weights mostly intact
  - Simple to implement and debug
  - Proven effective in ClipCap (CVPR workshop, 2021)
  - Only the projection layer + embedding layer need training
    initially, with optional full fine-tuning after
"""

import torch
import torch.nn as nn
from transformers import BioGptForCausalLM, BioGptTokenizer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.report_gen.projection import VisualProjection
from src.vae.vae import VAE


class MedReportGenerator(nn.Module):
    """
    Full report generation model: VAE (frozen) + Projection + BioGPT.

    Args:
        vae_checkpoint : path to trained VAE checkpoint
        num_tokens     : number of visual tokens (16)
        freeze_vae     : keep VAE encoder frozen during BioGPT training
                         (VAE already trained in Phase 3, no need to retrain)
    """

    def __init__(
        self,
        vae_checkpoint : str  = "checkpoints/vae/vae_best.pth",
        num_tokens     : int  = 16,
        freeze_vae     : bool = True,
    ):
        super().__init__()

        # ── Load pretrained BioGPT ─────────────────────────────────
        print("Loading BioGPT...")
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.biogpt = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        self.biogpt_dim = self.biogpt.config.hidden_size   # 768 for BioGPT

        # Freeze ALL BioGPT layers first
        for param in self.biogpt.parameters():
            param.requires_grad = False

        # Unfreeze only the last 2 transformer layers + output head
        # These are the layers that adapt to our medical report style
        # while earlier layers keep their pretrained medical knowledge
        for layer in self.biogpt.biogpt.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Always unfreeze the output projection (lm_head)
        for param in self.biogpt.output_projection.parameters():
                param.requires_grad = True

        # ── Load pretrained VAE (Phase 3) ───────────────────────────
        print(f"Loading VAE from {vae_checkpoint}...")
        self.vae = VAE(in_channels=1, latent_channels=256)
        ckpt = torch.load(vae_checkpoint, map_location="cpu")
        self.vae.load_state_dict(ckpt["model_state_dict"])

        if freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()
        self.freeze_vae = freeze_vae

        # ── Visual projection layer (this is what we actually train) ─
        self.projection = VisualProjection(
            latent_channels = 256,
            num_tokens       = num_tokens,
            biogpt_dim       = self.biogpt_dim,
        )
        self.num_tokens = num_tokens

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Image → visual tokens ready for BioGPT.

        Input:  image [B, 1, 256, 256]
        Output: visual_tokens [B, num_tokens, 768]
        """
        if self.freeze_vae:
            with torch.no_grad():
                z, mu, logvar = self.vae.encoder(image)
        else:
            z, mu, logvar = self.vae.encoder(image)

        visual_tokens = self.projection(z)   # [B, num_tokens, 768]
        return visual_tokens

    def forward(
        self,
        image      : torch.Tensor,   # [B, 1, 256, 256]
        input_ids  : torch.Tensor,   # [B, seq_len]  tokenized report
        labels     : torch.Tensor = None,  # [B, seq_len]  same as input_ids for LM loss
    ) -> dict:
        """
        Training forward pass.

        Steps:
          1. Encode image to visual tokens
          2. Embed text tokens using BioGPT's own embedding layer
          3. Concatenate: [visual_tokens, text_embeddings]
          4. Feed combined sequence through BioGPT
          5. Compute LM loss only on the TEXT portion (not visual tokens)
        """
        B = image.shape[0]

        # Step 1: image → visual tokens
        visual_tokens = self.encode_image(image)   # [B, num_tokens, 768]

        # Step 2: text tokens → embeddings (use BioGPT's own embedding table)
        text_embeds = self.biogpt.biogpt.embed_tokens(input_ids)   # [B, seq_len, 768]

        # Step 3: concatenate visual prefix + text embeddings
        combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        # shape: [B, num_tokens + seq_len, 768]

        # Step 4: build attention mask (all 1s — both visual and text are valid)
        visual_mask = torch.ones(B, self.num_tokens, device=image.device)
        text_mask   = (input_ids != self.tokenizer.pad_token_id).float()
        combined_mask = torch.cat([visual_mask, text_mask], dim=1)

        # Step 5: build labels — visual tokens get -100 (ignored in loss)
        if labels is not None:
            visual_labels = torch.full(
                (B, self.num_tokens), -100, dtype=torch.long, device=image.device
            )
            combined_labels = torch.cat([visual_labels, labels], dim=1)
        else:
            combined_labels = None

        # Step 6: forward through BioGPT using embeddings directly
        outputs = self.biogpt(
            inputs_embeds  = combined_embeds,
            attention_mask = combined_mask,
            labels         = combined_labels,
        )

        return outputs   # outputs.loss, outputs.logits available

    @torch.no_grad()
    def generate(
        self,
        image       : torch.Tensor,   # [1, 1, 256, 256]  single image
        max_length  : int = 128,
        num_beams   : int = 4,
    ) -> str:
        """
        Inference: image → generated report text.

        Uses BioGPT's built-in .generate() with our visual tokens
        as the starting context instead of text tokens.
        """
        self.eval()
        visual_tokens = self.encode_image(image)   # [1, num_tokens, 768]

        visual_mask = torch.ones(
            1, self.num_tokens, device=image.device
        )

        generated = self.biogpt.generate(
            inputs_embeds   = visual_tokens,
            attention_mask  = visual_mask,
            max_length      = max_length,
            num_beams       = num_beams,
            early_stopping  = True,
            pad_token_id    = self.tokenizer.pad_token_id,
        )

        report_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return report_text


# ── Quick Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing MedReportGenerator...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MedReportGenerator(
        vae_checkpoint = "checkpoints/vae/vae_best.pth",
        num_tokens     = 16,
        freeze_vae     = True,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters : {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Total parameters     : {total_params:,} ({total_params/1e6:.2f}M)")

    # Fake batch
    image     = torch.randn(2, 1, 256, 256).to(device)
    input_ids = torch.randint(0, 1000, (2, 32)).to(device)
    labels    = input_ids.clone()

    outputs = model(image, input_ids, labels)
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")

    print("[OK] MedReportGenerator working!")
    print("\nNext: src/report_gen/train.py")