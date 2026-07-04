# MedReportGen — My Project Context (for learning)

> Upload this file at the start of any Claude conversation to get full context.
> Then ask: "Explain Phase X to me from scratch" or "What does this file do?"

---

## Who I am
- Student at IIT Jodhpur
- Built this project but want to understand it deeply from scratch
- GitHub: HimanshuShekhar12
- Email: him62065kr@gmail.com

---

## What this project does (in plain English)

I built an AI system that looks at a chest X-ray image and automatically writes a radiology report — the kind a doctor would write. It also checks itself: it flags the report for human review if it's not confident, or if the report contradicts what the image classifier found.

**The core problem it solves:**
Medical datasets are imbalanced. For example, only 57 real images of fibrosis exist, but 500 are needed to train a good model. So I first trained a diffusion model (DDPM) to generate 443 fake-but-realistic fibrosis X-rays, then used the balanced dataset to train everything else.

---

## Tech stack
- **PyTorch** — everything is in PyTorch
- **HuggingFace Transformers** — BioGPT (the language model)
- **Gradio** — the web demo UI
- **OpenCV** — image loading and preprocessing
- **DenseNet121** (torchvision) — the disease classifier backbone

---

## Project location
`/data/b23_himanshu_shekhar/clip/pushed/repgenmed/`

GitHub repo: `HimanshuShekhar12/Medical-Report-Generator`

---

## The 7 Phases — what each one does

### Phase 1 — Data Pipeline (`src/data/`)
**What it does:** Downloads the OpenI chest X-ray dataset and prepares it.
- `download.py` — downloads images + XML reports
- `parse_reports.py` — reads XML files, extracts "findings" and "impression" text
- `preprocess.py` — resizes all images to 256×256, normalizes pixel values
- `dataset.py` — PyTorch Dataset class that loads images + reports for training

**Key concept to learn:** PyTorch `Dataset` and `DataLoader`

---

### Phase 2 — DDPM (`src/ddpm/`)
**What it does:** Generates synthetic chest X-rays using a diffusion model.

**Why needed:** Many disease classes have too few images. Fibrosis only has 57 real images. The DDPM generates realistic fake images of specific diseases to balance the dataset.

**How it works (simplified):**
1. Take a real X-ray
2. Slowly add random noise until it's pure noise (forward process, 1000 steps)
3. Train a neural network (UNet) to reverse this — remove noise step by step
4. At inference: start from pure noise, condition on a disease class → get a synthetic X-ray of that disease

**Files:**
- `diffusion.py` — the math: forward noising + reverse denoising
- `unet.py` — the neural network that predicts noise at each step
- `conditioning.py` — how the disease class label is injected (class embedding + timestep embedding)
- `train.py` — training loop
- `sample.py` — generates synthetic images per class
- `evaluate.py` — measures quality: MS-SSIM, FID, pathology preservation

**Key results:**
- All 13 rare classes balanced to 500 samples
- Pathology preservation > 0.3 for all classes (classifier recognises the target disease in generated images)
- Best val loss: 0.0159

**Key concepts to learn:** Diffusion models, UNet architecture, classifier-free guidance

---

### Phase 3 — VAE (`src/vae/`)
**What it does:** Learns a compressed representation of chest X-rays.

**How it works:**
- Input: X-ray image [1, 256, 256]
- Encoder compresses it to a small latent vector [256, 16, 16]
- Decoder reconstructs the image from the latent vector
- Trained with two losses: reconstruction loss (how similar is the output to input?) + KL divergence (keep the latent space well-organised)

**Why needed:** Instead of feeding the raw 256×256 image to BioGPT (which only understands text), we compress the image to a small latent vector and convert that to "visual tokens" BioGPT can read.

**Files:**
- `encoder.py` — CNN that compresses image → latent
- `decoder.py` — CNN that expands latent → image
- `vae.py` — combines encoder + decoder, computes KL loss
- `train.py` — training loop
- `evaluate.py` — reconstruction quality

**Key concepts to learn:** Variational Autoencoder (VAE), KL divergence, reparameterization trick, latent space

---

### Phase 4 — CheXNet Classifier (`src/classifier/`)
**What it does:** Looks at a chest X-ray and predicts which of 18 diseases are present.

**Architecture:**
- DenseNet121 backbone (pretrained on ImageNet)
- Replace the final layer with our own: 18 outputs, each 0-1 probability
- Sigmoid activation (not softmax) because a patient can have MULTIPLE diseases at once

**Two training runs (ablation study):**
1. **Baseline:** trained on real data only
2. **Augmented:** trained on real + DDPM synthetic data
→ Augmented version has higher F1 for rare classes — this is the proof that DDPM helped

**Files:**
- `chexnet.py` — model definition
- `train.py` — training loop with per-class binary cross-entropy loss
- `evaluate.py` — AUC, F1 per class, finds optimal thresholds, prints ablation table

**18 pathology classes:**
normal, effusion, pleural, pneumothorax, consolidation, infiltrate, opacity, atelectasis, edema, cardiomegaly, nodule, pneumonia, fracture, mass, calcification, emphysema, hernia, fibrosis

**Key concepts to learn:** DenseNet, multi-label classification, sigmoid vs softmax, AUC, F1 score, class imbalance

---

### Phase 5 — BioGPT Report Generation (`src/report_gen/`)
**What it does:** Generates the actual radiology report text from the X-ray image.

**Architecture (visual prefix approach):**
```
X-ray image [1, 256, 256]
  → VAE Encoder (frozen) → latent [256, 16, 16]
  → VisualProjection     → 32 visual tokens [32, 768]
  → prepend to BioGPT    → generates report text
```

**Why BioGPT?** It's a GPT model pretrained on 15 million biomedical papers and abstracts. It already "knows" medical language — we just fine-tune it to use image information.

**Visual prefix trick (ClipCap idea):**
- We don't modify BioGPT's internal architecture
- We convert the image into 32 "fake word tokens" and prepend them to the text
- BioGPT's attention naturally lets text tokens attend to these image tokens
- Only the projection layer + last 3 BioGPT layers are trained; rest stays frozen

**Important bug fixes made during training:**
1. Attention mask must be all-ones (not padding-based) — newer transformers versions compress output length otherwise
2. Loss computed manually with cross-entropy — newer transformers has a shape mismatch bug when labels are passed directly
3. num_tokens increased 16→32; projection uses 1D adaptive pool (2D pool breaks for non-square token counts)

**Uncertainty (MC Dropout):**
- Normally dropout is OFF during inference
- MC Dropout keeps it ON, runs 20 forward passes
- Each pass gives a slightly different report
- Measure pairwise ROUGE-L similarity across 20 reports
- Low similarity = model is uncertain → FLAG FOR REVIEW
- Threshold: uncertainty < 0.15 → AUTO-APPROVE

**Files:**
- `model.py` — MedReportGenerator (VAE + Projection + BioGPT)
- `projection.py` — VisualProjection: latent [256,16,16] → 32 tokens [32,768]
- `train.py` — fine-tuning loop
- `generate.py` — inference: image path → report text
- `evaluate.py` — BLEU, ROUGE metrics
- `uncertainty.py` — MC Dropout uncertainty estimation

**Key concepts to learn:** GPT / autoregressive language models, attention mechanism, fine-tuning vs pretraining, visual prefix, MC Dropout, ROUGE score

---

### Phase 6 — Full Pipeline (`src/pipeline/`)
**What it does:** Ties all phases together into one end-to-end system.

**Flow:**
1. Load image
2. Generate report (BioGPT)
3. Classify pathologies (CheXNet) → 18 probabilities
4. Consistency check: does the report text mention what the classifier found?
   - E.g. classifier says effusion=0.8 (positive), but report doesn't mention effusion → INCONSISTENT
5. Gate decision: CONSISTENT + low uncertainty → AUTO-APPROVE, else → FLAG FOR REVIEW

**Files:**
- `consistency_check.py` — keyword matching between report text and classifier output
- `run_pipeline.py` — CLI script that runs all stages on one image

**Run:**
```bash
python3 src/pipeline/run_pipeline.py --image_path <path>
python3 src/pipeline/run_pipeline.py --image_path <path> --uncertainty
```

---

### Phase 7 — Demo UI (`src/demo/app.py`)
**What it does:** Gradio web app for anyone to use the system without code.

**Features:**
- Upload chest X-ray → get report instantly
- Bar chart of all 18 pathology probabilities
- Consistency check table (report vs classifier per disease)
- Colour-coded gate badge: green AUTO-APPROVE / red FLAG FOR REVIEW
- Optional MC Dropout uncertainty panel
- PWA-installable (works as a mobile/desktop app)

**Run:**
```bash
python3 src/demo/app.py --share   # gives public URL
```

**For permanent hosting:** Deploy to HuggingFace Spaces using `spaces_app.py`
- Checkpoints stored in HF model repo, downloaded at startup
- Free CPU tier, $0.60/hr for GPU

---

## Checkpoint files (not on GitHub — gitignored)
```
checkpoints/
├── vae/vae_best.pth                        # trained VAE
├── report_gen/biogpt_best.pth              # fine-tuned BioGPT
└── classifier/chexnet_augmented_best.pth   # CheXNet (with synthetic data)
```

---

## Key design decisions (important for understanding)

| Decision | Why |
|----------|-----|
| DenseNet not ResNet | Dense connections: each layer sees ALL previous layers → better gradient flow for subtle medical features |
| Sigmoid not Softmax | Patient can have multiple diseases simultaneously — each class is independent |
| Visual prefix not cross-attention | Simpler, keeps BioGPT pretrained weights intact, proven by ClipCap paper |
| 1D pool in projection | 2D pool requires perfect square num_tokens (sqrt(32) is not integer), 1D works for any value |
| ROUGE-L not exact match for uncertainty | "No acute disease" and "No acute abnormality" are the same clinically but exact match gives 0% |
| Manual cross-entropy loss | Newer transformers has a bug: passing labels causes shape mismatch, manual CE avoids it |
| Frozen VAE during BioGPT training | VAE already trained; retraining it while training BioGPT would destabilize training |
| Only last 3 BioGPT layers unfrozen | Earlier layers hold general medical language knowledge; only last layers need to adapt to our task |

---

## How to ask Claude to help you learn

Say things like:
- "Explain how the VAE encoder works in `src/vae/encoder.py` line by line"
- "What is KL divergence and why do we use it in the VAE?"
- "Why does DDPM use 1000 timesteps? What happens if we use fewer?"
- "Explain the attention mask bug that was fixed in `src/report_gen/model.py`"
- "What is ROUGE-L and how is it used in `src/report_gen/uncertainty.py`?"
- "Walk me through what happens step by step when I run `run_pipeline.py`"
- "Explain DenseNet vs ResNet — why is DenseNet better for X-rays?"
- "What is classifier-free guidance in `src/ddpm/diffusion.py`?"

---

## Learning order (recommended)

1. PyTorch basics → understand `dataset.py`
2. CNNs → understand `encoder.py` and `chexnet.py`
3. VAE theory → understand `vae.py`
4. Diffusion models → understand `diffusion.py` and `unet.py`
5. Transformers + attention → understand `model.py` and `projection.py`
6. Fine-tuning + BioGPT → understand `train.py` in report_gen
7. Uncertainty estimation → understand `uncertainty.py`
8. Full system → understand `run_pipeline.py`
