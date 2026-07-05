# MedReportGen — Chest X-Ray Report Generator

**DDPM · VAE · BioGPT · CheXNet**

Generates structured radiology reports from chest X-ray images using a 4-stage AI pipeline.
Solves the rare-disease class imbalance problem by generating synthetic X-rays with a diffusion model,
then training the report generator and classifier on the balanced dataset.

**Live Demo:** [himanshushekhars/medreportgen on HuggingFace Spaces](https://huggingface.co/spaces/himanshushekhars/medreportgen)

---

> **Medical Disclaimer**
>
> This is a **research prototype only**. It is not a medical device and has not been validated
> for clinical use. Do **not** use this tool to diagnose, treat, or make any clinical decision
> about a patient. Always consult a qualified radiologist or physician.
> Results may be incorrect, incomplete, or misleading.

---

## Demo Screenshot

![MedReportGen Demo](assets/demo.png)

The demo takes a chest X-ray image and returns:
- A generated radiology report (FINDINGS + IMPRESSION)
- A bar chart of pathology probabilities for 18 classes
- A consistency table comparing what the report says vs what the classifier detects
- A final gate decision: **AUTO-APPROVE** (consistent + confident) or **FLAG FOR REVIEW** (mismatch or uncertain)

---

## How the Pipeline Works

```
Chest X-Ray (PNG)
       │
       ▼
┌──────────────┐    ┌───────────────────┐    ┌──────────────────────┐
│  VAE Encoder │───▶│ Visual Projection  │───▶│  BioGPT (fine-tuned) │
│  (frozen)    │    │  32 visual tokens  │    │  report generation   │
└──────────────┘    └───────────────────┘    └──────────┬───────────┘
                                                         │
                                               Generated Report Text
                                                         │
              ┌──────────────────────────────────────────┤
              │                                          │
              ▼                                          ▼
┌─────────────────────┐                  ┌──────────────────────────┐
│ CheXNet Classifier  │                  │  Keyword Consistency Gate │
│ 18-class sigmoid    │─────────────────▶│  report text vs probs    │
│ (DenseNet121)       │                  └──────────────┬───────────┘
└─────────────────────┘                                 │
                                         ┌──────────────▼───────────┐
                                         │  AUTO-APPROVE  ✅         │
                                         │  FLAG FOR REVIEW  ⚠️      │
                                         └──────────────────────────┘
```

### Stage 1 — VAE Encoder
The chest X-ray `[B, 1, 256, 256]` is passed through a trained Variational Autoencoder.
The encoder compresses it into a compact latent vector `[B, 256, 16, 16]`.
The VAE is frozen at this stage — we only use it as a feature extractor.

### Stage 2 — Visual Projection + BioGPT
The latent vector is passed through a trainable `VisualProjection` layer that maps it into
32 visual tokens `[B, 32, 1024]` — the same size as BioGPT word embeddings.
These tokens are prepended to the text input so BioGPT "sees" the image before generating text.
Only the last 3 BioGPT transformer layers and the output head are fine-tuned; the rest stays pretrained.

**Output:** A structured report with FINDINGS and IMPRESSION sections.

### Stage 3 — CheXNet Classifier
A DenseNet121 backbone with a multi-label sigmoid output head predicts probabilities for 18 pathology classes independently.
Sigmoid (not softmax) is used because a patient can have multiple conditions simultaneously.
Per-class thresholds are tuned to maximise F1.

**Output:** A probability vector — one score per pathology (0 to 1).

### Stage 4 — Consistency Gate
The report text is scanned for clinical keywords (e.g. "effusion", "pneumothorax", "consolidation").
These are compared against the classifier's predictions.
If the report mentions a condition but the classifier disagrees (or vice versa), the system flags it for human review.
Clinically critical mismatches (effusion, pneumothorax, cardiomegaly, etc.) always trigger the flag.

**Output:** AUTO-APPROVE or FLAG FOR REVIEW.

---

## Results

### DDPM Synthetic Data Generation

Many rare pathology classes have very few real training images (e.g. fibrosis = 57 images).
The DDPM generates synthetic X-rays to balance every class to 500 samples.

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

**Generation quality** (val_loss: 0.0159):

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

**What these metrics mean:**
- **MS-SSIM** — structural similarity to real images. Low values are partly expected because pairing is random.
- **FID** — distribution distance between real and synthetic (lower = more realistic). High FID means images are not photorealistic, but disease features are still preserved.
- **Pathology Prob** — the most clinically meaningful metric: does a pretrained classifier recognise the target disease in the generated image? All 13 classes score > 0.3, meaning the synthetic images carry the right pathology signal.

### Classifier Ablation (Baseline vs DDPM-Augmented)

The classifier was trained twice — once on real data only, once on real + synthetic data.
11 out of 24 rare-class metrics improved with synthetic augmentation.
The mixed results are honest: high FID scores mean some synthetic images are not photorealistic enough
to fool the classifier into learning better features for every class.

### Report Generation

BioGPT fine-tuned on the OpenI dataset generates FINDINGS + IMPRESSION text from the visual prefix.
The model is evaluated with BLEU and ROUGE-L scores against ground-truth radiology reports.
MC Dropout uncertainty estimation runs 20 stochastic passes and measures pairwise ROUGE-L similarity.
Low similarity across passes → high uncertainty → FLAG FOR REVIEW.

---

## Limitations

These are honest limitations of the current system. Understanding them is as important as understanding what it can do.

**1. Not photorealistic synthetic images**
The DDPM generates X-rays that preserve pathology signal (confirmed by classifier probe) but have high FID scores,
meaning they don't look like real X-rays at a pixel level. This limits the benefit of augmentation for some classes.

**2. Small dataset**
The OpenI dataset has ~7,400 images across ~3,955 patients. This is small by medical AI standards.
Models trained on small datasets may not generalise to images from different hospitals, scanners, or patient populations.

**3. Report quality is research-grade**
BioGPT generates grammatically reasonable reports but they may contain hallucinated findings or miss real ones.
The model has no grounding mechanism — it cannot point to which region of the image triggered each finding.
Do not interpret the report as a radiologist's opinion.

**4. Consistency gate is keyword-based**
The gate compares simple keyword matches between report text and classifier predictions.
It will miss subtle clinical language ("airspace opacity" vs "consolidation") and may raise false alarms.
It is a safety net, not a second opinion.

**5. CPU inference is slow**
On the free HuggingFace Spaces CPU tier, model loading takes ~2 minutes after the Space wakes from sleep (30 min idle).
Report generation takes ~10-20 seconds per image.

**6. Single view only**
The pipeline processes only frontal (PA) chest X-rays. Lateral views, CT scans, and other modalities are not supported.

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
| 7 | Gradio Demo UI — Deployed on HuggingFace Spaces | ✅ Complete |

---

## Quick Start (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run the demo (requires trained checkpoints in `checkpoints/`)
```bash
python3 src/demo/app.py
python3 src/demo/app.py --share     # public Gradio tunnel URL
python3 src/demo/app.py --port 7861
```

### 3. CLI — single image
```bash
python3 src/pipeline/run_pipeline.py --image_path data/processed/images/CXR1000_IM-0003-1001.png
```

### 4. CLI — with uncertainty estimation
```bash
python3 src/pipeline/run_pipeline.py --image_path <path> --uncertainty --json_out outputs/result.json
```

---

## Training Order

```bash
# Phase 1 — prepare data
python3 src/data/download.py
python3 src/data/preprocess.py

# Phase 2 — train DDPM, generate synthetic images
python3 src/ddpm/train.py
python3 src/ddpm/sample.py

# Phase 3 — train VAE
python3 src/vae/train.py

# Phase 4 — train classifier
python3 src/classifier/train.py --tag baseline
python3 src/classifier/train.py --tag augmented --use_balanced
python3 src/classifier/evaluate.py --tag augmented   # ablation table

# Phase 5 — fine-tune BioGPT
python3 src/report_gen/train.py

# Run demo
python3 src/demo/app.py --share
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
│   │   └── dataset.py           # PyTorch Dataset (18-class labels)
│   │
│   ├── ddpm/
│   │   ├── diffusion.py         # DDPM forward + reverse (linear + cosine schedule)
│   │   ├── conditioning.py      # timestep + class embeddings (18 classes)
│   │   ├── unet.py              # UNet backbone
│   │   ├── train.py             # DDPM training loop
│   │   ├── sample.py            # generate synthetic X-rays per class
│   │   └── evaluate.py          # MS-SSIM, FID, pathology preservation
│   │
│   ├── vae/
│   │   ├── encoder.py           # CNN encoder → latent [B, 256, 16, 16]
│   │   ├── decoder.py           # latent → reconstructed image
│   │   ├── vae.py               # VAE (KL divergence + reconstruction loss)
│   │   ├── train.py             # VAE training loop
│   │   └── evaluate.py          # reconstruction quality metrics
│   │
│   ├── classifier/
│   │   ├── chexnet.py           # DenseNet121 multi-label classifier (18 classes)
│   │   ├── train.py             # classifier training (baseline + augmented)
│   │   └── evaluate.py          # per-class AUC, F1, ablation table
│   │
│   ├── report_gen/
│   │   ├── model.py             # MedReportGenerator (VAE + Projection + BioGPT)
│   │   ├── projection.py        # VisualProjection: latent → 32 visual tokens
│   │   ├── train.py             # BioGPT fine-tuning loop
│   │   ├── generate.py          # inference: image → report text
│   │   ├── evaluate.py          # BLEU, ROUGE-L metrics
│   │   └── uncertainty.py       # MC Dropout uncertainty (ROUGE-L pairwise)
│   │
│   ├── pipeline/
│   │   ├── consistency_check.py # keyword match: report text vs classifier probs
│   │   └── run_pipeline.py      # end-to-end CLI: image → report → gate
│   │
│   └── demo/
│       └── app.py               # Gradio web UI (lazy model loading)
│
├── spaces_app.py                # HuggingFace Spaces entry point (downloads checkpoints)
├── assets/
│   └── demo.png                 # demo screenshot
├── checkpoints/                 # trained weights (gitignored — too large)
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
- 18 pathology labels: normal, effusion, pleural, pneumothorax, infiltrate, cardiomegaly,
  consolidation, atelectasis, opacity, edema, pneumonia, nodule, mass, fracture,
  calcification, emphysema, hernia, fibrosis
- Source: https://openi.nlm.nih.gov

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| DenseNet121 for classifier | Dense connections improve gradient flow — important for subtle medical texture features |
| Sigmoid not Softmax | Patients commonly have multiple pathologies simultaneously |
| Visual prefix (not cross-attention) | Keeps BioGPT weights intact; cross-attention would require architectural changes that break pretraining |
| 1D adaptive pool in projection | 2D pool requires a perfect-square num_tokens; sqrt(32) is not an integer — 1D pool avoids this |
| ROUGE-L for uncertainty | Exact string match is too strict; "No acute disease" and "No acute abnormality" are clinically identical |
| Manual cross-entropy loss | Newer transformers versions have a shape-mismatch bug when passing labels directly to the model |
| Freeze VAE encoder | The VAE is trained separately and used as a frozen feature extractor — fine-tuning it jointly would destabilise BioGPT training |
| Only last 3 BioGPT layers unfrozen | Balances adaptation to radiology language vs catastrophic forgetting of pretrained biomedical knowledge |
