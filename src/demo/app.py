"""
app.py
------
Gradio demo UI for MedReportGen end-to-end pipeline.

X-ray image → BioGPT report → CheXNet classification → consistency gate.

Run from project root:
  python3 src/demo/app.py
  python3 src/demo/app.py --share        # public URL via Gradio tunnel
  python3 src/demo/app.py --port 7861    # custom port
"""

import sys
import os
import argparse
import torch
import matplotlib
matplotlib.use("Agg")   # headless — must come before pyplot import
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import gradio as gr

from src.report_gen.model import MedReportGenerator
from src.report_gen.generate import load_image_as_tensor
from src.report_gen.uncertainty import mc_dropout_generate
from src.classifier.chexnet import CheXNet
from src.data.dataset import XRayDataset
from src.pipeline.consistency_check import check_consistency

# ── Constants ───────────────────────────────────────────────────────────────

LABEL_NAMES     = list(XRayDataset.LABEL_MAP.keys())   # 18 ordered pathology names
VAE_CKPT        = os.environ.get("VAE_CKPT",        "checkpoints/vae/vae_best.pth")
BIOGPT_CKPT     = os.environ.get("BIOGPT_CKPT",     "checkpoints/report_gen/biogpt_best.pth")
CLASSIFIER_CKPT = os.environ.get("CLASSIFIER_CKPT", "checkpoints/classifier/chexnet_augmented_best.pth")
NUM_TOKENS      = 16

# ── Lazy model cache ────────────────────────────────────────────────────────
# Models are heavy (~1.5 GB combined). Load once on first inference click,
# then keep in memory for all subsequent calls.

_cache: dict = {}


def _get_models():
    if "report" not in _cache:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _cache["device"] = device

        print(f"[demo] Loading models on {device}...")

        # Report generator (VAE + BioGPT)
        gen = MedReportGenerator(
            vae_checkpoint=VAE_CKPT,
            num_tokens=NUM_TOKENS,
            freeze_vae=True,
        ).to(device)
        ckpt = torch.load(BIOGPT_CKPT, map_location=device)
        gen.projection.load_state_dict(ckpt["projection_state_dict"])
        gen.biogpt.load_state_dict(ckpt["biogpt_state_dict"])
        gen.eval()
        _cache["report"] = gen

        # CheXNet multi-label classifier
        clf = CheXNet(num_classes=len(LABEL_NAMES)).to(device)
        clf_ckpt = torch.load(CLASSIFIER_CKPT, map_location=device)
        clf.load_state_dict(clf_ckpt["model_state_dict"])
        clf.eval()
        _cache["classifier"] = clf

        print("[demo] Models ready.")

    return _cache["report"], _cache["classifier"], _cache["device"]


# ── Rendering helpers ────────────────────────────────────────────────────────

def _prob_chart(probs: dict) -> plt.Figure:
    """Horizontal bar chart, sorted ascending so highest prob is at top."""
    pairs  = sorted(probs.items(), key=lambda x: x[1])   # ascending for barh
    names  = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors = ["#e74c3c" if v >= 0.5 else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(8, 10))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.65)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability", fontsize=11)
    ax.set_title("CheXNet Pathology Probabilities", fontsize=13,
                 fontweight="bold", pad=12)
    ax.axvline(x=0.5, color="#888", linestyle="--", linewidth=1,
               alpha=0.6, label="threshold = 0.5")
    ax.legend(fontsize=9, loc="lower right")

    for bar, val in zip(bars, values):
        ax.text(
            min(val + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=8, color="#333",
        )

    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")
    plt.tight_layout()
    return fig


def _consistency_html(consistency: dict,
                      uncertainty_score,
                      uncertainty_status) -> str:
    status = consistency["status"]
    hdr_bg = "#27ae60" if status == "CONSISTENT" else "#e74c3c"

    rows = []
    for name, d in sorted(
        consistency["details"].items(), key=lambda x: -x[1]["classifier"]
    ):
        row_bg    = "#fff5f5" if not d["agree"] else "white"
        clf_color = "#e74c3c" if d["clf_positive"] else "#7f8c8d"
        clf_label = "POS" if d["clf_positive"] else "NEG"
        rep_label = "yes" if d["report_says"] else "no"
        agree_ico = "✅" if d["agree"] else "❌"
        rows.append(f"""
          <tr style="background:{row_bg}">
            <td style="padding:5px 10px;font-weight:600">{name}</td>
            <td style="padding:5px 10px;text-align:center">{rep_label}</td>
            <td style="padding:5px 10px;text-align:center">
              <span style="color:{clf_color};font-weight:700">{clf_label}</span>
              <span style="color:#999;font-size:11px"> {d['classifier']:.3f}</span>
            </td>
            <td style="padding:5px 10px;text-align:center">{agree_ico}</td>
          </tr>""")

    unc_block = ""
    if uncertainty_score is not None:
        uc = "#e74c3c" if uncertainty_status == "FLAG FOR REVIEW" else "#27ae60"
        unc_block = f"""
        <div style="margin-top:12px;padding:10px 14px;background:#f0f0f0;
                    border-radius:6px;font-family:sans-serif;font-size:13px">
          <b>MC Dropout Uncertainty:</b>
          <span style="color:{uc};font-weight:700;margin-left:8px">
            {uncertainty_score:.3f} — {uncertainty_status}
          </span>
        </div>"""

    return f"""
    <div style="font-family:sans-serif;font-size:13px">
      <div style="padding:10px 14px;background:{hdr_bg};color:white;
                  border-radius:6px;font-weight:700;font-size:15px;
                  margin-bottom:10px">
        Consistency: {status}
      </div>
      <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse">
          <thead>
            <tr style="background:#eeeeee">
              <th style="padding:6px 10px;text-align:left">Pathology</th>
              <th style="padding:6px 10px">Report</th>
              <th style="padding:6px 10px">Classifier</th>
              <th style="padding:6px 10px">Agree</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
      {unc_block}
    </div>"""


def _gate_html(gate: str) -> str:
    color = "#27ae60" if gate == "AUTO-APPROVE" else "#c0392b"
    icon  = "✅" if gate == "AUTO-APPROVE" else "⚠️"
    return f"""
    <div style="margin-top:8px;padding:18px 24px;background:{color};
                color:white;border-radius:8px;font-family:sans-serif;
                text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.15)">
      <div style="font-size:30px">{icon}</div>
      <div style="font-size:20px;font-weight:700;margin-top:6px">{gate}</div>
    </div>"""


# ── Core inference function ─────────────────────────────────────────────────

def run_inference(image_path: str, use_uncertainty: bool, mc_passes: int):
    """
    Called by Gradio on each button click.

    Returns:
      report_text      : str
      prob_figure      : matplotlib Figure
      consistency_html : str (HTML)
      gate_html        : str (HTML)
    """
    if image_path is None:
        return (
            "Please upload a chest X-ray image.",
            None,
            "<p style='color:gray;font-family:sans-serif'>Upload an image first.</p>",
            "<p style='color:gray;font-family:sans-serif'>—</p>",
        )

    try:
        gen_model, clf_model, device = _get_models()

        image = load_image_as_tensor(image_path).to(device)   # [1,1,256,256]

        # Stage 1 — report generation
        uncertainty_score  = None
        uncertainty_status = None

        if use_uncertainty:
            unc = mc_dropout_generate(
                gen_model, image,
                num_passes=int(mc_passes),
                max_length=128,
            )
            report_text        = unc["most_common_report"]
            uncertainty_score  = round(unc["uncertainty_score"], 4)
            uncertainty_status = unc["status"]
        else:
            report_text = gen_model.generate(image, max_length=128, num_beams=4)

        # Stage 2 — CheXNet classification
        with torch.no_grad():
            raw_probs = clf_model(image).squeeze(0).cpu().tolist()   # [18]
        classifier_probs = {
            name: round(raw_probs[i], 4) for i, name in enumerate(LABEL_NAMES)
        }

        # Stage 3 — consistency check
        consistency = check_consistency(report_text, classifier_probs)

        # Stage 4 — final gate
        if consistency["status"] == "INCONSISTENT":
            gate = "FLAG FOR REVIEW"
        elif uncertainty_status == "FLAG FOR REVIEW":
            gate = "FLAG FOR REVIEW"
        else:
            gate = "AUTO-APPROVE"

        return (
            report_text,
            _prob_chart(classifier_probs),
            _consistency_html(consistency, uncertainty_score, uncertainty_status),
            _gate_html(gate),
        )

    except Exception as exc:
        msg = f"[ERROR] {type(exc).__name__}: {exc}"
        err_html = f"<p style='color:red;font-family:monospace'>{msg}</p>"
        return msg, None, err_html, err_html


# ── Gradio layout ───────────────────────────────────────────────────────────

CSS = """
.report-box textarea {
    font-family: Georgia, serif !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
}
"""

DESCRIPTION = """
# 🏥 MedReportGen — Chest X-Ray Report Generator
**DDPM · VAE · BioGPT · CheXNet** — Upload a frontal chest X-ray (PA view, PNG) to generate
a structured radiology report, pathology confidence scores, and a consistency-gated review decision.
"""

PIPELINE_INFO = """
---
**Pipeline stages**
1. **VAE encoder** → compact latent representation of the X-ray
2. **BioGPT** (visual prefix) → structured FINDINGS + IMPRESSION report
3. **CheXNet** (DenseNet121) → 18-class pathology probability vector
4. **Keyword consistency gate** → compares report language against classifier predictions
5. **Final decision** → `AUTO-APPROVE ✅` (consistent + confident) or `FLAG FOR REVIEW ⚠️`
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MedReportGen") as demo:
        gr.Markdown(DESCRIPTION)

        # ── Row 1: inputs + primary outputs ──────────────────────────────
        with gr.Row(equal_height=False):

            # Left column — upload + settings
            with gr.Column(scale=1, min_width=260):
                image_input = gr.Image(
                    type="filepath",
                    label="Chest X-Ray (PA view, PNG)",
                    height=280,
                )

                with gr.Accordion("Options", open=False):
                    uncertainty_cb = gr.Checkbox(
                        label="MC Dropout Uncertainty Estimation",
                        value=False,
                        info=(
                            "Runs multiple stochastic forward passes to measure "
                            "confidence. Slower — useful for borderline cases."
                        ),
                    )
                    mc_passes = gr.Slider(
                        minimum=10, maximum=50, value=20, step=5,
                        label="MC Dropout Passes",
                        visible=False,
                    )
                    uncertainty_cb.change(
                        fn=lambda x: gr.Slider(visible=x),
                        inputs=uncertainty_cb,
                        outputs=mc_passes,
                    )

                submit_btn = gr.Button("Generate Report", variant="primary", size="lg")

            # Right column — report text + gate badge
            with gr.Column(scale=2):
                report_output = gr.Textbox(
                    label="Generated Radiology Report",
                    lines=9,
                    elem_classes=["report-box"],
                    placeholder="Generated report will appear here after clicking 'Generate Report'.",
                )
                gate_output = gr.HTML()

        # ── Row 2: chart + consistency table ─────────────────────────────
        gr.Markdown("---")
        with gr.Row(equal_height=False):
            prob_output        = gr.Plot()
            consistency_output = gr.HTML(label="Consistency Check")

        # ── Button click handler ──────────────────────────────────────────
        submit_btn.click(
            fn=run_inference,
            inputs=[image_input, uncertainty_cb, mc_passes],
            outputs=[report_output, prob_output, consistency_output, gate_output],
        )

        gr.Markdown(PIPELINE_INFO)

    return demo


# ── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedReportGen Gradio Demo")
    parser.add_argument("--share",  action="store_true", help="Create public Gradio URL")
    parser.add_argument("--port",   type=int, default=7860,      help="Port to serve on")
    parser.add_argument("--server", type=str, default="0.0.0.0", help="Bind address")
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        css=CSS,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
