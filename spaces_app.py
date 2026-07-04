"""
spaces_app.py
-------------
Entry point for Hugging Face Spaces deployment.

On Spaces, checkpoints are downloaded from your HF model repo
at first launch and cached under /tmp/checkpoints/.

Set the following Spaces SECRET (Settings → Variables and secrets):
  HF_MODEL_REPO  — e.g. "HimanshuShekhar12/medreportgen-checkpoints"
"""

import os
import sys
from pathlib import Path

# ── Download checkpoints from HF Hub ────────────────────────────────────────
from huggingface_hub import hf_hub_download

HF_REPO  = os.environ.get("HF_MODEL_REPO", "HimanshuShekhar12/medreportgen-checkpoints")
CKPT_DIR = Path("/tmp/checkpoints")

FILES = {
    "vae/vae_best.pth"                           : "vae_best.pth",
    "report_gen/biogpt_best.pth"                 : "biogpt_best.pth",
    "classifier/chexnet_augmented_best.pth"      : "chexnet_augmented_best.pth",
}

def _download_checkpoints():
    for local_rel, hf_filename in FILES.items():
        local_path = CKPT_DIR / local_rel
        if local_path.exists():
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[spaces] Downloading {hf_filename} from {HF_REPO}...")
        src = hf_hub_download(repo_id=HF_REPO, filename=hf_filename)
        import shutil
        shutil.copy(src, local_path)
        print(f"[spaces] Saved → {local_path}")

_download_checkpoints()

# Point the demo at the /tmp checkpoint paths
os.environ["VAE_CKPT"]        = str(CKPT_DIR / "vae/vae_best.pth")
os.environ["BIOGPT_CKPT"]     = str(CKPT_DIR / "report_gen/biogpt_best.pth")
os.environ["CLASSIFIER_CKPT"] = str(CKPT_DIR / "classifier/chexnet_augmented_best.pth")

# ── Launch the demo ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.demo.app import build_demo, CSS
import gradio as gr

demo = build_demo()
demo.launch(css=CSS, theme=gr.themes.Soft())
