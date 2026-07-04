"""
run_pipeline.py
---------------
End-to-end chest X-ray → radiology report pipeline.

Stages:
  1. MedReportGenerator (VAE + BioGPT) → report text
  2. CheXNet multi-label classifier    → per-pathology probabilities
  3. Consistency check                 → CONSISTENT / INCONSISTENT
  4. (Optional) MC Dropout uncertainty → uncertainty score
  5. Final gate                        → AUTO-APPROVE / FLAG FOR REVIEW

Run on a single image:
  python3 src/pipeline/run_pipeline.py --image_path <path>

With MC Dropout uncertainty:
  python3 src/pipeline/run_pipeline.py --image_path <path> --uncertainty

Save JSON output:
  python3 src/pipeline/run_pipeline.py --image_path <path> --json_out outputs/result.json
"""

import sys
import argparse
import json
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.report_gen.model import MedReportGenerator
from src.report_gen.generate import load_image_as_tensor
from src.report_gen.uncertainty import mc_dropout_generate
from src.classifier.chexnet import CheXNet
from src.data.dataset import XRayDataset
from src.pipeline.consistency_check import check_consistency

LABEL_NAMES = list(XRayDataset.LABEL_MAP.keys())   # ordered list of 18 pathology names


def get_config():
    parser = argparse.ArgumentParser(description="End-to-end CXR report generation pipeline")
    parser.add_argument("--vae_checkpoint",        type=str, default="checkpoints/vae/vae_best.pth")
    parser.add_argument("--biogpt_checkpoint",     type=str, default="checkpoints/report_gen/biogpt_best.pth")
    parser.add_argument("--classifier_checkpoint", type=str, default="checkpoints/classifier/chexnet_augmented_best.pth")
    parser.add_argument("--num_tokens",            type=int, default=16)
    parser.add_argument("--max_length",            type=int, default=128)
    parser.add_argument("--num_beams",             type=int, default=4)
    parser.add_argument("--image_path",            type=str, required=True,  help="Path to input chest X-ray (.png)")
    parser.add_argument("--uncertainty",           action="store_true",      help="Run MC Dropout uncertainty estimation")
    parser.add_argument("--mc_passes",             type=int, default=20,     help="Number of MC Dropout forward passes")
    parser.add_argument("--gpu",                   type=int, default=1)
    parser.add_argument("--json_out",              type=str, default=None,   help="Write pipeline result to this JSON path")
    return parser.parse_args()


# ── Model loaders ───────────────────────────────────────────────────────────

def _load_report_model(config, device) -> MedReportGenerator:
    model = MedReportGenerator(
        vae_checkpoint=config.vae_checkpoint,
        num_tokens=config.num_tokens,
        freeze_vae=True,
    ).to(device)

    ckpt = torch.load(config.biogpt_checkpoint, map_location=device)
    model.projection.load_state_dict(ckpt["projection_state_dict"])
    model.biogpt.load_state_dict(ckpt["biogpt_state_dict"])
    model.eval()
    return model


def _load_classifier(config, device) -> CheXNet:
    model = CheXNet(num_classes=len(LABEL_NAMES)).to(device)
    ckpt  = torch.load(config.classifier_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── Classifier inference ────────────────────────────────────────────────────

@torch.no_grad()
def _classify(classifier: CheXNet, image: torch.Tensor) -> dict:
    """Returns {pathology_name: probability} for all 18 classes."""
    probs = classifier(image).squeeze(0).cpu().tolist()   # [18] floats
    return {name: round(probs[i], 4) for i, name in enumerate(LABEL_NAMES)}


# ── Pretty-print helpers ────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(title)
    print("="*60)


def _print_probs(probs: dict) -> None:
    for name, p in sorted(probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 20)
        print(f"  {name:<16} {p:.3f}  {bar}")


# ── Main pipeline ───────────────────────────────────────────────────────────

def run_pipeline(config, device: torch.device) -> dict:
    """
    Full end-to-end pipeline for one X-ray image.

    Returns a dict with keys:
      image_path, report, classifier_probs, consistency,
      uncertainty_score, gate_status
    """
    _print_section("MedReportGen — End-to-End Pipeline")
    print(f"Image : {config.image_path}")
    print(f"Device: {device}")

    # ── Load models ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading report generator (VAE + BioGPT)...")
    report_model = _load_report_model(config, device)
    print("[OK]")

    print("[2/4] Loading CheXNet classifier...")
    classifier = _load_classifier(config, device)
    print("[OK]")

    # ── Load image ───────────────────────────────────────────────────────────
    image = load_image_as_tensor(config.image_path).to(device)   # [1,1,256,256]

    # ── Stage 1 : Report generation ──────────────────────────────────────────
    print("\n[3/4] Generating radiology report...")

    uncertainty_score  = None
    uncertainty_status = None

    if config.uncertainty:
        print(f"       (MC Dropout — {config.mc_passes} passes)")
        unc_result         = mc_dropout_generate(
            report_model, image,
            num_passes = config.mc_passes,
            max_length = config.max_length,
        )
        report_text        = unc_result["most_common_report"]
        uncertainty_score  = round(unc_result["uncertainty_score"], 4)
        uncertainty_status = unc_result["status"]
    else:
        report_text = report_model.generate(
            image,
            max_length = config.max_length,
            num_beams  = config.num_beams,
        )
    print("[OK]")

    # ── Stage 2 : Classification ─────────────────────────────────────────────
    print("[4/4] Running CheXNet classifier...")
    classifier_probs = _classify(classifier, image)
    print("[OK]")

    # ── Stage 3 : Consistency gate ───────────────────────────────────────────
    consistency = check_consistency(report_text, classifier_probs)

    # ── Stage 4 : Final gate decision ────────────────────────────────────────
    if consistency["status"] == "INCONSISTENT":
        gate_status = "FLAG FOR REVIEW"
    elif uncertainty_status == "FLAG FOR REVIEW":
        gate_status = "FLAG FOR REVIEW"
    else:
        gate_status = "AUTO-APPROVE"

    # ── Print results ────────────────────────────────────────────────────────
    _print_section("GENERATED REPORT")
    print(report_text)

    _print_section("PATHOLOGY PROBABILITIES (CheXNet)")
    _print_probs(classifier_probs)

    _print_section("CONSISTENCY CHECK")
    print(f"Status              : {consistency['status']}")
    if consistency["mismatches"]:
        print(f"Mismatches          : {', '.join(consistency['mismatches'])}")
    if consistency["clinical_mismatches"]:
        print(f"Clinical mismatches : {', '.join(consistency['clinical_mismatches'])}")

    if uncertainty_score is not None:
        _print_section("UNCERTAINTY (MC Dropout)")
        print(f"Uncertainty score   : {uncertainty_score:.3f}")
        print(f"Status              : {uncertainty_status}")

    _print_section(f"FINAL STATUS: {gate_status}")

    # ── Build result dict ────────────────────────────────────────────────────
    result = {
        "image_path"       : config.image_path,
        "report"           : report_text,
        "classifier_probs" : classifier_probs,
        "consistency"      : consistency,
        "uncertainty_score": uncertainty_score,
        "gate_status"      : gate_status,
    }

    if config.json_out:
        out_path = Path(config.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nOutput saved → {out_path}")

    return result


def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_pipeline(config, device)


if __name__ == "__main__":
    main()
