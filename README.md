# MedReportGen
Multimodal Medical Report Generator using DDPM + VAE + BioGPT

## Architecture
- DDPM: Synthetic X-ray generation for rare disease classes
- VAE: Image encoding to latent space
- BioGPT: Report generation with uncertainty estimation

## Progress
- Phase 1: Data Pipeline (done)
- Phase 2: DDPM Training (in progress)
- Phase 3: VAE
- Phase 4: Classifier
- Phase 5: Report Generation
- Phase 6: Full Pipeline
- Phase 7: Demo UI

## Dataset
OpenI Indiana University Chest X-ray Dataset
- 7,470 images, 3,955 reports

## Setup
pip install -r requirements.txt
pip install -e .