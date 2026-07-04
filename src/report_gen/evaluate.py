"""
evaluate.py
-----------
Computes BLEU and ROUGE scores for the report generator over the full test set.

Run:
  python3 src/report_gen/evaluate.py
"""

import sys
import argparse
import torch
import csv
from pathlib import Path
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer as rouge_lib

nltk.download("punkt_tab", quiet=True)

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.report_gen.generate import load_model
from src.data.dataset import XRayReportDataset, get_dataloader


def get_config():
    parser = argparse.ArgumentParser(description="Evaluate report generator on test set")
    parser.add_argument("--vae_checkpoint",    type=str, default="checkpoints/vae/vae_best.pth")
    parser.add_argument("--biogpt_checkpoint", type=str, default="checkpoints/report_gen/biogpt_best.pth")
    parser.add_argument("--num_tokens",        type=int, default=32)
    parser.add_argument("--max_length",        type=int, default=128)
    parser.add_argument("--num_beams",         type=int, default=4)
    parser.add_argument("--gpu",               type=int, default=0)
    parser.add_argument("--output_csv",        type=str, default="outputs/evaluation_metrics.csv")
    return parser.parse_args()


def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model on {device}...")
    model = load_model(config, device)
    print("[OK] Model loaded\n")

    test_loader = get_dataloader(
        XRayReportDataset,
        split        = "test",
        batch_size   = 1,
        num_workers  = 0,
        use_balanced = False,
    )
    print(f"Test samples: {len(test_loader)}\n")

    scorer   = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    smoother = SmoothingFunction().method1

    references  = []   # for BLEU: list of [list of ref tokens]
    hypotheses  = []   # for BLEU: list of hyp tokens
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    rows = []

    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        image = batch["image"].to(device)

        generated = model.generate(
            image,
            max_length = config.max_length,
            num_beams  = config.num_beams,
        )

        ground_truth = model.tokenizer.decode(
            batch["input_ids"][0], skip_special_tokens=True
        )

        # ROUGE
        rouge = scorer.score(ground_truth, generated)
        rouge1_scores.append(rouge["rouge1"].fmeasure)
        rouge2_scores.append(rouge["rouge2"].fmeasure)
        rougeL_scores.append(rouge["rougeL"].fmeasure)

        # BLEU (tokenize by whitespace)
        ref_tokens = ground_truth.lower().split()
        hyp_tokens = generated.lower().split()
        references.append([ref_tokens])
        hypotheses.append(hyp_tokens)

        rows.append({
            "sample"      : i + 1,
            "ground_truth": ground_truth,
            "generated"   : generated,
            "rouge1"      : round(rouge["rouge1"].fmeasure, 4),
            "rouge2"      : round(rouge["rouge2"].fmeasure, 4),
            "rougeL"      : round(rouge["rougeL"].fmeasure, 4),
        })

    # Corpus-level BLEU
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoother)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

    avg_r1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_r2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rL = sum(rougeL_scores) / len(rougeL_scores)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Samples evaluated : {len(test_loader)}")
    print(f"BLEU-1            : {bleu1:.4f}")
    print(f"BLEU-2            : {bleu2:.4f}")
    print(f"BLEU-4            : {bleu4:.4f}")
    print(f"ROUGE-1           : {avg_r1:.4f}")
    print(f"ROUGE-2           : {avg_r2:.4f}")
    print(f"ROUGE-L           : {avg_rL:.4f}")
    print("=" * 50)

    # Save per-sample results to CSV
    out_path = Path(config.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPer-sample results saved → {out_path}")


if __name__ == "__main__":
    main()
