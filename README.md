# MedReportGen
**Multimodal Medical Report Generator — DDPM + VAE + BioGPT**

Generates structured radiology reports from chest X-ray images.
Solves the rare-disease class imbalance problem by first generating synthetic X-rays
using a diffusion model, then training the report generator on the balanced dataset.

---

## How it works (big picture)

```
Chest X-Ray Image
       │
       ▼
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  VAE Encoder│────▶│ Visual Projection │────▶│  BioGPT (fine-   │
│  (Phase 3)  │     │    (num_tokens=32)│     │  tuned on CXR)   │
└─────────────┘     └──────────────────┘     └────────┬──────────┘
                                                       │
                                              Generated Report Text
                                                       │
       ┌───────────────────────────────────────────────┤
       │                                               │
       ▼                                               ▼
┌─────────────────┐                       ┌────────────────────────┐
│ CheXNet (Phase 4)│                      │  Consistency Check     │
│ 18-class probs   │─────────────────────▶│  (Phase 6)             │
└─────────────────┘                       └────────────┬───────────┘
                                                       │
                                          ┌────────────▼───────────┐
                                          │  AUTO-APPROVE  ✅       │
                                          │  FLAG FOR REVIEW ⚠️     │
                                          └────────────────────────┘
```

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Pipeline | ✅ Complete |
| 2 | DDPM — Synthetic X-Ray Generation | ✅ Complete |
| 3 | VAE — Image Encoder | ✅ Complete |
| 4 | CheXNet — Pathology Classifier | ✅ Complete |
| 5 | BioGPT — Report Generation | ✅ Complete |
| 6 | End-to-End Pipeline + Consistency Gate | ✅ Complete |
| 7 | Gradio Demo UI (PWA-ready) | ✅ Complete |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run the demo (requires trained checkpoints)
```bash
# Local demo
python3 src/demo/app.py

# Share publicly (1-week link)
python3 src/demo/app.py --share

# Custom port
python3 src/demo/app.py --port 7861
```

### 3. Run on a single image (CLI)
```bash
python3 src/pipeline/run_pipeline.py --image_path data/processed/images/CXR1000_IM-0003-1001.png
```

### 4. With uncertainty estimation
```bash
python3 src/pipeline/run_pipeline.py --image_path <path> --uncertainty --json_out outputs/result.json
```

---

## Project Structure

```
repgenmed/
├── src/
│   ├── data/
│   │   ├── download.py          # download OpenI dataset
│   │   ├── parse_reports.py     # XML reports → findings + impression
│   │   ├── preprocess.py        # resize, normalize images
│   │   ├── augment.py           # augmentation (horizontal flip)
│   │   └── dataset.py           # PyTorch Dataset classes
│   │
│   ├── ddpm/
│   │   ├── diffusion.py         # DDPM forward + reverse process (linear + cosine schedule)
│   │   ├── conditioning.py      # timestep + class embeddings (18 classes)
│   │   ├── unet.py              # UNet backbone
│   │   ├── train.py             # DDPM training loop
│   │   ├── sample.py            # generate synthetic X-rays per class
│   │   └── evaluate.py          # MS-SSIM, FID, pathology preservation
│   │
│   ├── vae/
│   │   ├── encoder.py           # CNN encoder → latent [B, 256, 16, 16]
│   │   ├── decoder.py           # latent → reconstructed image
│   │   ├── vae.py               # VAE (KL loss + reconstruction loss)
│   │   ├── train.py             # VAE training loop
│   │   └── evaluate.py          # reconstruction quality metrics
│   │
│   ├── classifier/
│   │   ├── chexnet.py           # DenseNet121 multi-label classifier (18 classes)
│   │   ├── train.py             # classifier training (baseline + augmented)
│   │   └── evaluate.py          # per-class AUC, F1, ablation study table
│   │
│   ├── report_gen/
│   │   ├── model.py             # MedReportGenerator (VAE + Projection + BioGPT)
│   │   ├── projection.py        # VisualProjection: latent → 32 visual tokens
│   │   ├── train.py             # BioGPT fine-tuning
│   │   ├── generate.py          # inference: image → report text
│   │   ├── evaluate.py          # BLEU, ROUGE metrics
│   │   └── uncertainty.py       # MC Dropout uncertainty estimation
│   │
│   ├── pipeline/
│   │   ├── consistency_check.py # compare report text vs classifier predictions
│   │   └── run_pipeline.py      # end-to-end CLI: image → report → gate decision
│   │
│   └── demo/
│       └── app.py               # Gradio web UI
│
├── spaces_app.py                # HuggingFace Spaces entry point
├── checkpoints/                 # trained model weights (gitignored)
├── outputs/                     # logs, eval results (gitignored)
├── data/                        # dataset (gitignored)
├── requirements.txt
└── setup.py
```

---

## Dataset

**OpenI Indiana University Chest X-ray Dataset**
- 7,430 images across 3,955 patient studies
- Frontal (PA) + lateral views
- XML radiology reports with findings and impression sections
- Source: https://openi.nlm.nih.gov

---

## Phase Details

### Phase 1 — Data Pipeline
- Parse XML reports → extract `findings` and `impression` sections
- Assign 18 pathology labels per study
- Split into train / val / test

### Phase 2 — DDPM (Synthetic Data Generation)

**Why?** Many pathology classes have very few real samples (e.g. fibrosis = 57 images).
The DDPM generates synthetic X-rays for each rare class to balance the dataset to 500 samples per class.

**Architecture:** UNet with timestep + class conditioning, 1000 diffusion steps

| Class | Real | Synthetic | Total |
|-------|------|-----------|-------|
| fibrosis | 57 | 443 | 500 |
| hernia | 103 | 397 | 500 |
| emphysema | 242 | 258 | 500 |
| calcification | 328 | 172 | 500 |
| mass | 367 | 133 | 500 |
| fracture | 395 | 105 | 500 |
| nodule | 352 | 148 | 500 |
| edema | 388 | 112 | 500 |
| cardiomegaly | 359 | 141 | 500 |
| atelectasis | 444 | 56 | 500 |
| infiltrate | 418 | 82 | 500 |
| opacity | 394 | 106 | 500 |

**Final DDPM Results** (val_loss: 0.0159, base_channels=32, linear schedule):

| Class | MS-SSIM | FID | Pathology Prob | Status |
|-------|---------|-----|----------------|--------|
| atelectasis | 0.211 | 358.3 | 0.670 | ✅ PASS |
| calcification | 0.191 | 309.0 | 0.369 | ✅ PASS |
| cardiomegaly | 0.220 | 347.9 | 0.583 | ✅ PASS |
| edema | 0.132 | 294.6 | 0.467 | ✅ PASS |
| emphysema | 0.165 | 313.7 | 0.447 | ✅ PASS |
| fibrosis | 0.182 | 338.3 | 0.377 | ✅ PASS |
| fracture | 0.230 | 340.3 | 0.609 | ✅ PASS |
| hernia | 0.201 | 359.1 | 0.470 | ✅ PASS |
| infiltrate | 0.225 | 284.5 | 0.568 | ✅ PASS |
| mass | 0.215 | 320.8 | 0.415 | ✅ PASS |
| nodule | 0.188 | 320.4 | 0.473 | ✅ PASS |
| opacity | 0.207 | 329.1 | 0.490 | ✅ PASS |
| pneumonia | 0.203 | 513.5 | 0.537 | ✅ PASS |

> Pathology Prob > 0.3 means a pretrained CheXNet classifier recognises the target disease in the generated image — all 13 classes pass.

**What the metrics mean:**
- **MS-SSIM** — structural similarity to real images (higher = more similar). Low scores are partly expected due to random pairing.
- **FID** — distribution gap between real and synthetic (lower = better). High FID means images aren't photorealistic, but pathology features are preserved.
- **Pathology Prob** — most clinically meaningful: does the generated image actually look like the target disease?

### Phase 3 — VAE (Variational Autoencoder)
- Encodes X-ray images into a compact latent space: `[B, 1, 256, 256]` → `[B, 256, 16, 16]`
- Trained with reconstruction loss + KL divergence
- The encoder is frozen and reused in the report generation pipeline

### Phase 4 — CheXNet Classifier
- DenseNet121 backbone with multi-label sigmoid output (18 classes)
- Trained twice: **baseline** (real data only) vs **augmented** (real + synthetic)
- Per-class threshold tuning to maximise F1
- Ablation study proves DDPM synthetic data improves rare-class F1

### Phase 5 — BioGPT Report Generation

**Architecture (visual prefix approach):**
```
Image [B,1,256,256]
  → VAE Encoder (frozen)     → latent [B, 256, 16, 16]
  → VisualProjection (1D pool) → 32 visual tokens [B, 32, 768]
  → prepend to BioGPT input  → report text
```

**Key implementation details:**
- Only last 3 BioGPT transformer layers + output head are fine-tuned (rest stays pretrained)
- Attention mask is all-ones (avoids transformers output-length compression bug)
- Loss computed manually with cross-entropy (avoids transformers shape-mismatch bug when passing labels directly)
- Generation uses `repetition_penalty=1.3` and `min_new_tokens=20` to prevent degenerate output

**Uncertainty estimation (MC Dropout):**
- Runs 20 stochastic forward passes with dropout active at inference time
- Measures pairwise ROUGE-L similarity across passes (higher similarity = more certain)
- Uncertainty score < 0.15 → AUTO-APPROVE; else → FLAG FOR REVIEW

### Phase 6 — End-to-End Pipeline
```bash
python3 src/pipeline/run_pipeline.py --image_path <path>
```
1. Load image → generate report (BioGPT)
2. Run CheXNet → get 18-class probabilities
3. Compare report keywords vs classifier predictions
4. Clinical mismatches (effusion, pneumothorax, cardiomegaly, etc.) → FLAG FOR REVIEW
5. Consistent + low uncertainty → AUTO-APPROVE

### Phase 7 — Demo UI
Gradio web app with:
- X-ray image upload
- Generated report text
- Pathology probability bar chart (18 classes)
- Consistency check table (report vs classifier per pathology)
- Colour-coded gate badge (green = AUTO-APPROVE, red = FLAG FOR REVIEW)
- Optional MC Dropout uncertainty panel

PWA-installable — works as a mobile/desktop app directly from the browser.

**Deploy to HuggingFace Spaces (permanent hosting):**
```bash
# 1. Upload checkpoints to HF Hub
huggingface-cli repo create medreportgen-checkpoints --type model
# upload vae_best.pth, biogpt_best.pth, chexnet_augmented_best.pth

# 2. Create Space
huggingface-cli repo create medreportgen --type space --space_sdk gradio
# copy src/, spaces_app.py, requirements.txt → push to Space
```

---

## Training (run in order)

```bash
# Phase 1 — prepare data
python3 src/data/download.py
python3 src/data/preprocess.py

# Phase 2 — train DDPM and generate synthetic images
python3 src/ddpm/train.py
python3 src/ddpm/sample.py

# Phase 3 — train VAE
python3 src/vae/train.py

# Phase 4 — train classifier (baseline, then augmented)
python3 src/classifier/train.py --tag baseline
python3 src/classifier/train.py --tag augmented --use_balanced
python3 src/classifier/evaluate.py --tag augmented   # prints ablation table

# Phase 5 — fine-tune BioGPT
python3 src/report_gen/train.py

# Run demo
python3 src/demo/app.py --share
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| DenseNet121 for classifier | Dense connections improve gradient flow for subtle medical features |
| Sigmoid not Softmax | Patients can have multiple pathologies simultaneously |
| Visual prefix (not cross-attention) | Keeps BioGPT weights intact; simpler; proven by ClipCap |
| 1D adaptive pool in projection | Avoids square-root constraint on num_tokens (2D pool breaks for non-perfect-squares like 32) |
| ROUGE-L for uncertainty | Exact string match is too strict; "No acute disease" and "No acute abnormality" are clinically identical |
| Manual cross-entropy loss | Newer transformers versions have a shape-mismatch bug when labels are passed directly |
