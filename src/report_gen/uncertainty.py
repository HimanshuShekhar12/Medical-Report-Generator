"""
uncertainty.py
--------------
Monte Carlo Dropout uncertainty estimation for generated reports.

Why this matters clinically:
  A language model can generate a confident-sounding report
  even when it's completely wrong — this is the "AI overconfidence"
  problem. MC Dropout gives us a way to measure how UNCERTAIN
  the model actually is, so low-confidence reports can be flagged
  for radiologist review instead of being auto-approved.

How MC Dropout works:
  Normally, dropout is turned OFF during inference (.eval() mode)
  so the model gives a single deterministic output. MC Dropout
  keeps dropout ON during inference and runs the SAME input
  through the model N times. Because dropout randomly zeroes
  different neurons each pass, you get N slightly different
  outputs. The VARIANCE across these N outputs is your
  uncertainty estimate — high variance = low confidence.

Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation"

Run:
  python3 src/report_gen/uncertainty.py --image_path <path>
"""

import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.report_gen.model import MedReportGenerator
from src.report_gen.generate import load_model, load_image_as_tensor, get_config as base_get_config


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """
    Forces all Dropout layers to stay in training mode (active)
    even though the rest of the model is in eval mode.

    Why not just call model.train()?
      That would also re-enable BatchNorm's running-stats update
      and other training-only behaviors we don't want during
      inference. We only want dropout layers to be stochastic.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_generate(
    model      : MedReportGenerator,
    image      : torch.Tensor,
    num_passes : int = 20,
    max_length : int = 128,
) -> dict:
    """
    Runs generation N times with dropout active to estimate
    token-level uncertainty.

    Returns:
      reports          : list of N generated report strings
      mean_token_probs : average max-probability per generated
                         token position (confidence proxy)
      uncertainty_score: 0-1 score, higher = less certain
    """
    model.eval()                 # base eval mode (no BatchNorm updates etc.)
    enable_mc_dropout(model)     # but force dropout layers back on

    reports = []
    all_logit_sequences = []

    for _ in range(num_passes):
        generated_ids = model.biogpt.generate(
            inputs_embeds  = model.encode_image(image),
            attention_mask = torch.ones(1, model.num_tokens, device=image.device),
            max_length     = max_length,
            num_beams      = 1,        # greedy for speed across N passes
            do_sample      = False,
            pad_token_id   = model.tokenizer.pad_token_id,
            output_scores          = True,
            return_dict_in_generate = True,
        )

        text = model.tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True)
        reports.append(text)

    # ── Agreement-based uncertainty ───────────────────────────────
    # Simple, robust approach: how often does the MOST COMMON
    # generated report match across the N passes?
    # High agreement → low uncertainty. Low agreement → high uncertainty.
    from collections import Counter
    counts = Counter(reports)
    most_common_report, most_common_count = counts.most_common(1)[0]
    agreement_ratio = most_common_count / num_passes

    uncertainty_score = 1.0 - agreement_ratio   # 0 = fully certain, 1 = fully uncertain

    return {
        "reports"           : reports,
        "most_common_report": most_common_report,
        "agreement_ratio"   : agreement_ratio,
        "uncertainty_score" : uncertainty_score,
        "status"            : "AUTO-APPROVE" if uncertainty_score < 0.3 else "FLAG FOR REVIEW",
    }


def main():
    parser = argparse.ArgumentParser(description="MC Dropout uncertainty estimation")
    parser.add_argument("--vae_checkpoint",    type=str, default="checkpoints/vae/vae_best.pth")
    parser.add_argument("--biogpt_checkpoint", type=str, default="checkpoints/report_gen/biogpt_best.pth")
    parser.add_argument("--num_tokens",        type=int, default=16)
    parser.add_argument("--image_path",        type=str, required=True)
    parser.add_argument("--num_passes",        type=int, default=20)
    parser.add_argument("--max_length",        type=int, default=128)
    parser.add_argument("--gpu",               type=int, default=1)
    config = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model on {device}...")
    model = load_model(config, device)
    print("[OK] Model loaded\n")

    image = load_image_as_tensor(config.image_path).to(device)

    print(f"Running {config.num_passes} MC Dropout passes...\n")
    result = mc_dropout_generate(
        model, image,
        num_passes = config.num_passes,
        max_length = config.max_length,
    )

    print("="*55)
    print("MC Dropout Uncertainty Report")
    print("="*55)
    print(f"Most common report : {result['most_common_report'][:200]}")
    print(f"Agreement ratio    : {result['agreement_ratio']:.2%}")
    print(f"Uncertainty score  : {result['uncertainty_score']:.3f}")
    print(f"Status             : {result['status']}")
    print("\nAll generated variants:")
    for i, r in enumerate(result["reports"]):
        print(f"  [{i+1:2d}] {r[:100]}")


if __name__ == "__main__":
    main()