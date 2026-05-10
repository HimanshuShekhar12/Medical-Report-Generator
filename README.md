# MedReportGen
Multimodal Medical Report Generator using DDPM + VAE + BioGPT

Addresses the rare disease class imbalance problem in chest X-ray datasets
by generating synthetic X-rays using diffusion models before training the
report generation pipeline.

---

## Architecture

### Phase 1 — Data Pipeline
- Source: OpenI Indiana University Chest X-ray Dataset
- 7,430 raw images across 3,955 studies (frontal PA + lateral views)
- XML radiology reports parsed into findings + impression sections
- Labels extracted per study: 13 rare disease classes + normal
- Class balancing target: 500 samples per class

### Phase 2 — DDPM (Denoising Diffusion Probabilistic Model)
- Generates synthetic chest X-rays conditioned on disease class
- Trained on processed real images with class conditioning
- Best val loss: 0.0162 (early stopping at convergence)
- Generation targets per rare class (2,156 total synthetic images):

| Class         | Real | Synthetic | Total |
|---------------|------|-----------|-------|
| fibrosis      |   57 |       443 |   500 |
| hernia        |  103 |       397 |   500 |
| emphysema     |  242 |       258 |   500 |
| calcification |  328 |       172 |   500 |
| mass          |  367 |       133 |   500 |
| fracture      |  395 |       105 |   500 |
| nodule        |  352 |       148 |   500 |
| pneumonia     |  497 |         3 |   500 |
| edema         |  388 |       112 |   500 |
| cardiomegaly  |  359 |       141 |   500 |
| atelectasis   |  444 |        56 |   500 |
| infiltrate    |  418 |        82 |   500 |
| opacity       |  394 |       106 |   500 |

- Status: Sampling in progress → evaluation pending

### Phase 3 — VAE (Variational Autoencoder)
- Encodes X-ray images into compact latent representations
- Latent vectors passed to report generation model
- Status: Pending

### Phase 4 — Classifier
- Disease classification head for uncertainty estimation
- Status: Pending

### Phase 5 — Report Generation (BioGPT)
- Generates radiology reports from image latent + class label
- Evaluated with BLEU, ROUGE, CheXbert F1
- Status: Pending

### Phase 6 — Full Pipeline
- End-to-end: X-ray image → structured radiology report
- Status: Pending

### Phase 7 — Demo UI
- Gradio or Streamlit interface for inference
- Status: Pending

---

## Dataset
OpenI Indiana University Chest X-ray Dataset (NLMCXR)
- 7,430 images across 3,955 patient studies
- Frontal (PA) and lateral views
- XML radiology reports with findings and impression sections
- Source: https://openi.nlm.nih.gov

---

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Project Structure

```
repgenmed/
├── src/
│   ├── data/
│   │   ├── parse_reports.py     # XML → structured labels
│   │   ├── preprocess.py        # image normalization + resize
│   │   ├── augment.py           # augmentation pipeline
│   │   └── dataset.py           # PyTorch dataset class
│   └── ddpm/
│       ├── diffusion.py         # DDPM forward + reverse process
│       ├── conditioning.py      # class conditioning mechanism
│       ├── train.py             # training loop
│       ├── sample.py            # inference + image generation
│       └── evaluate.py          # MS-SSIM + FID evaluation
├── checkpoints/                 # saved model weights (gitignored)
├── outputs/                     # logs + eval results (gitignored)
└── data/                        # dataset (gitignored)
```