# Code Walkthrough — Every File Explained in Order

Read this file top to bottom. Each file is explained in the exact order
you would run the project. By the end you will know what every line is
for and why it exists.

---

# PHASE 1 — DATA PIPELINE
> Goal: Turn raw downloaded files into clean tensors ready for training.

---

## `src/data/download.py`

**What it does:** Downloads the OpenI chest X-ray dataset from the internet.

**Why it exists:** You can't train on data you don't have. This script
automates fetching 7,430 X-ray images and their XML report files so
anyone who clones the repo can reproduce the project.

**Key logic:**
- Downloads a ZIP file from openi.nlm.nih.gov
- Extracts images into `data/raw/images/`
- Extracts XML report files into `data/raw/reports/`

**Input:**  Internet connection
**Output:** `data/raw/` folder with images and XML files

**Connects to:** `parse_reports.py` and `preprocess.py` which read these raw files

---

## `src/data/parse_reports.py`

**What it does:** Reads the XML report files and extracts structured information.

**Why it exists:** The radiology reports are stored as XML like this:
```xml
<report>
  <finding>The heart is enlarged. Left effusion noted.</finding>
  <impression>Cardiomegaly with effusion.</impression>
  <MeSH>
    <major>Cardiomegaly</major>
    <major>Pleural Effusion</major>
  </MeSH>
</report>
```
We need to pull out: findings text, impression text, and disease labels.
Raw XML is useless for training — this file converts it into a clean CSV.

**Key logic:**
- Opens each XML file using Python's `xml.etree.ElementTree`
- Extracts `<finding>` tag → findings text
- Extracts `<impression>` tag → impression text
- Extracts `<MeSH>` tags → disease labels (maps to our 18-class label system)
- Writes one row per image to `data/processed/labels.csv`

**Input:**  `data/raw/reports/*.xml`
**Output:** `data/processed/labels.csv`
```
report_id, image_id, findings, impression, labels, split
3462, CXR3462_IM-1683-1001.png, "The heart is...", "No acute...", normal|hernia, train
```

**Connects to:** `preprocess.py` uses the image_id column to find which images to process.
`dataset.py` reads this CSV as its main data source.

---

## `src/data/preprocess.py`

**What it does:** Resizes all X-ray images to 256×256 and normalizes them.

**Why it exists:** Raw X-ray images come in different sizes — some 512×512,
some 1024×1024, some rectangular. Neural networks need fixed-size inputs.
Also, pixel values (0–255) need to be normalized to (-1, 1) for stable training.

**Key logic:**
```
Raw image (any size, any range)
    → cv2.resize(256, 256)         make all images same size
    → pixel / 255.0                scale to [0, 1]
    → pixel * 2 - 1                scale to [-1, 1]
    → save as PNG to data/processed/images/
```

**Why (-1, 1) not (0, 1)?**
Neural networks train better when inputs are centered around zero.
With (0,1), the average pixel value is ~0.5 — the network always needs
to subtract this mental baseline. With (-1,1), the average is ~0 and
gradients flow more cleanly.

**Input:**  `data/raw/images/*.png` (various sizes)
**Output:** `data/processed/images/*.png` (all 256×256, pixel range [-1,1])

**Connects to:** `dataset.py` loads images from this processed folder.

---

## `src/data/augment.py`

**What it does:** Defines augmentation transforms applied during training.

**Why it exists:** With limited data, augmentation artificially increases
variety. If you show the same 57 fibrosis images every epoch, the model
memorises them (overfits). If you randomly flip them horizontally, it
sees 114 effectively different images.

**Key logic:**
- Horizontal flip only (NOT vertical — upside-down lungs don't exist)
- Applied randomly during training (50% chance per image)
- NOT applied during validation/test (we want consistent evaluation)

**Why only horizontal flip for X-rays?**
Chest X-rays are symmetric left-right. A flipped X-ray is still medically
valid. But a vertically flipped X-ray (lungs at bottom, diaphragm at top)
never exists in real medicine — it would teach the model wrong anatomy.

**Input:**  Image tensor
**Output:** Randomly flipped image tensor

**Connects to:** `dataset.py` calls augmentation during `__getitem__` when `split="train"`

---

## `src/data/dataset.py`

**What it does:** The central PyTorch Dataset class that every training
script uses to load data.

**Why it exists:** PyTorch's DataLoader needs a Dataset object that
implements `__len__` (how many samples?) and `__getitem__` (give me
sample number N). This file defines three Dataset classes:

**Three classes inside:**

### `XRayDataset`
Used by: DDPM training, VAE training, CheXNet training
```
Returns per sample:
  image     → tensor [1, 256, 256]   the X-ray
  label     → int                    primary disease class (for DDPM conditioning)
  multi_hot → tensor [18]            all diseases as 0/1 (for CheXNet)
  report_id → string                 links to the report text
```

### `XRayReportDataset`
Used by: BioGPT training
```
Returns per sample:
  image     → tensor [1, 256, 256]   the X-ray
  input_ids → tensor [128]           tokenized report text
  labels    → tensor [128]           same as input_ids (for language model loss)
```

### `get_dataloader()`
A factory function. Instead of writing DataLoader setup code in every
training script, you call:
```python
loader = get_dataloader(XRayDataset, split="train", batch_size=16)
```

**Key design in `__getitem__`:**
```python
# Load image from disk
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# Normalize
img = img.float() / 255.0 * 2.0 - 1.0   # [-1, 1]
# Add channel dimension
img = img.unsqueeze(0)    # [256,256] → [1,256,256]
```

**Input:**  `data/processed/labels.csv` + image files
**Output:** Batches of (image, label) tensors for training scripts

**Connects to:** Every training script imports and uses this.

---

# PHASE 2 — DDPM (Synthetic Data Generation)
> Goal: Generate synthetic chest X-rays for rare disease classes.

---

## `src/ddpm/conditioning.py`

**What it does:** Converts the disease class label and timestep into
embedding vectors that guide the UNet during denoising.

**Why it exists:** The UNet needs two pieces of information at every step:
1. "What step of the denoising process are we at?" (timestep)
2. "What disease are we supposed to be generating?" (class label)

Both are integers. Neural networks can't use raw integers — they need
continuous vector representations. This file converts them.

**Two classes inside:**

### `TimestepEmbedding`
Converts timestep integer (0–1000) into a 256-dim vector.
Uses **sinusoidal positional encoding** — the same idea as in transformers.
Why sinusoidal? It encodes both magnitude AND frequency, so step 500
and step 501 are similar but distinguishable.
```
t = 437
→ sin(437 / 10000^(0/256)), cos(437 / 10000^(0/256)),
   sin(437 / 10000^(2/256)), cos(437 / 10000^(2/256)), ...
→ vector of 256 numbers that uniquely represent "step 437"
```

### `ClassEmbedding`
Converts class integer (0=normal, 1=effusion, ..., 17=fibrosis)
into a 256-dim learnable vector. Like a word embedding — the model
learns what "fibrosis" means visually during training.

### `Conditioning.forward()`
```python
return t_emb + c_emb   # [B, 256] combined conditioning signal
```
Simple addition. The UNet receives this at every block.

**Input:**  timestep integer, class index integer
**Output:** Combined conditioning vector [B, 256]

**Connects to:** `unet.py` injects this vector into every block via
addition to intermediate feature maps.

---

## `src/ddpm/diffusion.py`

**What it does:** Contains the DDPM mathematics — the forward noising
process and the reverse denoising process.

**Why it exists:** This is the core math of diffusion models. Separated
from the UNet so the noise schedule and sampling logic can be changed
independently of the network architecture.

**Key class: `DDPM`**

### `_linear_beta_schedule()` and `_cosine_beta_schedule()`
The noise schedule — how much noise to add at each step.

```
Linear:  β increases uniformly from 0.0001 to 0.02
         step 1: add tiny noise
         step 500: add medium noise
         step 1000: add large noise

Cosine:  β follows a cosine curve
         starts very slow, accelerates in middle, slows at end
         Often produces better quality but can overfit on small data
```

We use linear for final results because cosine with base_channels=64
caused our model to output pure noise (v2 failure).

### `forward_process()` — adding noise
```python
# The math:
x_t = sqrt(alpha_cumprod_t) * x_0  +  sqrt(1 - alpha_cumprod_t) * noise

# In plain English:
# x_t = (how much original image to keep) * original
#      + (how much noise to add) * random_noise
```
alpha_cumprod_t decreases from 1.0 (step 0, clean) to ~0 (step 1000, pure noise).

### `sample()` — the reverse process (inference)
```python
for t in reversed(range(1000)):
    # Predict noise using UNet
    predicted_noise = unet(x_t, t, class_idx)

    # Classifier-free guidance: amplify the class signal
    uncond_noise = unet(x_t, t, null_class)
    guided_noise = uncond_noise + scale * (predicted_noise - uncond_noise)

    # Remove the guided noise → one step cleaner
    x_{t-1} = remove_noise(x_t, guided_noise, t)
```

**Why classifier-free guidance?**
Without it, the model hedges — generates a generic X-ray. Guidance
amplifies the difference between "generate anything" and "generate fibrosis"
by scale=3.0, pushing the output strongly toward the target disease.

**Input:**  Clean X-ray image + class label (during training)
           Pure noise + class label (during sampling)
**Output:** Noisy images (training) / clean synthetic images (sampling)

**Connects to:** `unet.py` is called inside `sample()` to predict noise.
`train.py` calls `forward_process()` to create training pairs.

---

## `src/ddpm/unet.py`

**What it does:** The neural network that learns to predict noise at
each denoising step.

**Why UNet specifically:**
UNet was originally designed for medical image segmentation (2015).
Its key feature — skip connections — makes it perfect for denoising:

```
ENCODER (going down):
  [1, 256, 256] → [64, 128, 128] → [128, 64, 64] → [256, 32, 32]
  Each step: captures bigger, more abstract features
  Conditioning vector added at each level

BOTTLENECK:
  [256, 32, 32] → full global understanding of the image

DECODER (going up):
  [256, 32, 32] → [128, 64, 64] → [64, 128, 128] → [1, 256, 256]
  Each step: rebuilds spatial detail
  SKIP CONNECTION: receives encoder output at same scale

SKIP CONNECTION (the key):
  Encoder layer output ──────────────────┐
                                         ▼
  Bottleneck output → upsampled → concatenate → decoder block
```

Without skip connections, the decoder has to reconstruct fine spatial
detail from the compressed bottleneck alone. With them, it can access
the original spatial detail directly from the encoder.

**Input:**  Noisy image [B, 1, 256, 256] + conditioning vector [B, 256]
**Output:** Predicted noise [B, 1, 256, 256] — same shape as input

**Connects to:** Called by `diffusion.py`'s `sample()` at every denoising step.

---

## `src/ddpm/train.py`

**What it does:** The training loop for the DDPM.

**Why it exists:** Neural networks learn by seeing examples repeatedly
and adjusting weights. This script shows the UNet thousands of (noisy
image, true noise) pairs and trains it to predict the noise.

**Training loop logic:**
```
For each batch of real X-rays:
    1. Pick random timestep t for each image (0–1000)
    2. Generate random Gaussian noise ε
    3. Apply forward_process: x_t = add_noise(image, ε, t)
    4. UNet predicts: ε_pred = unet(x_t, t, class_label)
    5. Loss = MSE(ε_pred, ε)   ← how wrong was the noise prediction?
    6. Backpropagate → update UNet weights
```

**Why MSE loss for noise prediction?**
We're doing regression (predict a continuous noise vector), not
classification. MSE penalises large errors more than small ones —
encouraging the model to get the noise direction right.

**Input:**  Batches from `XRayDataset`
**Output:** `checkpoints/ddpm/ddpm_best.pth`

**Connects to:** `sample.py` loads this checkpoint to generate synthetic images.

---

## `src/ddpm/sample.py`

**What it does:** Uses the trained DDPM to generate synthetic X-rays
for each rare disease class.

**Why it exists:** Training proved the model can learn noise patterns.
Now we actually generate the images we need to balance the dataset.

**Key logic:**
```python
for class_name in RARE_CLASSES:
    needed = 500 - real_count[class_name]   # how many synthetic to make?

    # Generate in batches
    for batch in range(0, needed, batch_size):
        samples = ddpm.sample(
            class_idx=class_tensor,          # "generate fibrosis"
            image_size=256,
            guidance_scale=3.0,
        )
        save_images(samples, output_dir)
```

**Input:**  `checkpoints/ddpm/ddpm_best.pth`
**Output:** `data/balanced/images/` — new synthetic PNG files
            `data/balanced/final_labels.csv` — updated CSV including synthetic images

**Connects to:** `dataset.py` has a `use_balanced=True` option that reads
from `data/balanced/` instead of `data/processed/`. Classifier training
uses this balanced dataset.

---

## `src/ddpm/evaluate.py`

**What it does:** Measures the quality of generated synthetic images.

**Three metrics:**

### MS-SSIM (Multi-Scale Structural Similarity)
Compares a synthetic image to a real image.
Range: 0 (completely different) to 1 (identical).
Low scores don't necessarily mean failure — a synthetic fibrosis image
compared to a random real atelectasis image will naturally score low.

### FID (Fréchet Inception Distance)
Compares the DISTRIBUTION of synthetic images to the distribution of
real images using deep features from InceptionV3.
Lower = better. Our scores (284–513) are high — images are not photorealistic,
but that's partly expected with 57 training samples.

### Pathology Preservation
Runs a pretrained CheXNet on synthetic images and checks if the
target disease probability is > 0.3.
This is the most important metric — does the synthetic image actually
look like the target disease to a classifier?

**Input:**  Generated images + real images
**Output:** Printed table of MS-SSIM, FID, pathology prob per class

---

# PHASE 3 — VAE (Image Encoder)
> Goal: Compress X-ray images into a compact latent vector for BioGPT.

---

## `src/vae/encoder.py`

**What it does:** A CNN that compresses a 256×256 X-ray image into
a small latent representation.

**Architecture:**
```
Input [1, 256, 256]
  → Conv(1→32) + ReLU           [32, 256, 256]
  → Conv(32→64) + stride 2      [64, 128, 128]   ← downsample
  → Conv(64→128) + stride 2     [128, 64, 64]    ← downsample
  → Conv(128→256) + stride 2    [256, 32, 32]    ← downsample
  → Conv(256→256) + stride 2    [256, 16, 16]    ← downsample

Then split into TWO outputs:
  → Linear → μ (mean)    [256, 16, 16]
  → Linear → logσ² (log variance) [256, 16, 16]
```

**Why two outputs (μ and logσ²)?**
This is the VAE difference from a regular autoencoder.
Instead of mapping image → single fixed point, we map image → distribution.
μ is the centre, σ is the spread (uncertainty).
We then SAMPLE from N(μ, σ) to get z.

**Why logσ² instead of σ directly?**
σ must be positive. logσ² can be any real number — easier for the
network to output. We exponentiate it: σ = exp(0.5 × logσ²).

**Input:**  Image tensor [B, 1, 256, 256]
**Output:** μ [B, 256, 16, 16] and logσ² [B, 256, 16, 16]

**Connects to:** `vae.py` uses both outputs to sample z via reparameterization.

---

## `src/vae/decoder.py`

**What it does:** The reverse of encoder — expands latent vector back
to a full image.

**Architecture:**
```
Input z [256, 16, 16]
  → ConvTranspose(256→128) + stride 2  [128, 32, 32]   ← upsample
  → ConvTranspose(128→64) + stride 2   [64, 64, 64]    ← upsample
  → ConvTranspose(64→32) + stride 2    [32, 128, 128]  ← upsample
  → ConvTranspose(32→1) + stride 2     [1, 256, 256]   ← upsample
  → Tanh                               values in [-1, 1]
```

**Why Tanh at the end?**
Our images are normalised to [-1, 1]. Tanh squashes output to this
same range — ensures the reconstructed image is comparable to the input.

**IMPORTANT:** The decoder is only used during VAE training to compute
reconstruction loss. At inference time (BioGPT pipeline), we only use
the encoder. The decoder is discarded.

**Input:**  Latent z [B, 256, 16, 16]
**Output:** Reconstructed image [B, 1, 256, 256]

---

## `src/vae/vae.py`

**What it does:** Combines encoder + decoder and defines the VAE loss.

**The reparameterization trick (critical concept):**
```python
# We want: z = sample from N(μ, σ)
# Problem: can't backpropagate through a random sample

# Solution:
ε = torch.randn_like(μ)    # random noise (no gradients needed here)
z = μ + σ * ε              # gradients flow through μ and σ normally
```

This is the mathematical trick that makes VAEs trainable.
The randomness is in ε, which we don't need to differentiate.
Gradients flow through the deterministic path: μ and σ.

**The two losses:**
```python
# 1. Reconstruction loss — does output look like input?
recon_loss = MSE(reconstructed_image, original_image)

# 2. KL divergence — keep latent space organised
# Forces: μ → 0, σ → 1 for all images
kl_loss = -0.5 * sum(1 + logσ² - μ² - exp(logσ²))

# Total
loss = recon_loss + β * kl_loss
```

**What KL divergence does in plain English:**
Without it, the encoder maps each image to a specific isolated point.
Two fibrosis images might map to completely different locations.
Points between them decode to garbage.

With KL loss, ALL images are pulled toward the origin with spread ~1.
The latent space becomes a smooth, continuous ball where nearby points
decode to visually similar images. This organised structure is what
allows the projection layer to extract meaningful visual tokens for BioGPT.

**Input:**  Image [B, 1, 256, 256]
**Output:** (reconstructed image, μ, logσ², z, loss)

---

## `src/vae/train.py`

**What it does:** Trains the VAE — shows it images, asks it to reconstruct
them, penalises bad reconstructions and disorganised latent spaces.

**Training loop:**
```
For each batch:
    1. Feed image to encoder → get μ, logσ²
    2. Sample z via reparameterization trick
    3. Feed z to decoder → get reconstruction
    4. Compute recon_loss + KL_loss
    5. Backpropagate → update encoder + decoder weights
```

**Early stopping:** If validation loss doesn't improve for 10 epochs,
stop training. Prevents overfitting.

**Input:**  `XRayDataset` batches
**Output:** `checkpoints/vae/vae_best.pth`

**Connects to:** `model.py` in report_gen loads the encoder from this checkpoint.

---

## `src/vae/evaluate.py`

**What it does:** Checks if the VAE learned good reconstructions.

Loads the trained VAE, runs test images through encode→decode, and
measures reconstruction quality. Visual inspection (save side-by-side
comparison images) + quantitative SSIM score.

---

# PHASE 4 — CHEXNET CLASSIFIER
> Goal: Multi-label disease classification from X-ray images.

---

## `src/classifier/chexnet.py`

**What it does:** Defines the CheXNet model — DenseNet121 with a
custom 18-class multi-label head.

**Full architecture:**
```
Input [B, 1, 256, 256]   grayscale X-ray
  ↓
Repeat channel 3x → [B, 3, 256, 256]    DenseNet expects RGB
  ↓
DenseNet121 backbone (pretrained on ImageNet)
  Dense block 1 → Dense block 2 → Dense block 3 → Dense block 4
  (each layer receives ALL previous layers' outputs concatenated)
  ↓
Global Average Pool → [B, 1024]
  ↓
Dropout(0.4)
  ↓
Linear(1024 → 18)
  ↓
Sigmoid → [B, 18]    18 independent probabilities in [0, 1]
```

**What is a Dense Block?**
In a regular CNN:   layer N receives only layer N-1's output
In DenseNet:        layer N receives layers 1, 2, 3, ..., N-1 ALL concatenated

This means every layer has direct access to the original image features
AND all intermediate features. Gradients flow directly to every layer —
no vanishing gradient problem.

**Why pretrained on ImageNet if our images are medical?**
ImageNet pretrained weights know edges, textures, curves, and shapes.
An X-ray lung has edges and textures too. Starting from pretrained weights
gives us a huge head start versus random initialisation, especially with
our small dataset.

**Why Dropout(0.4) before the final layer?**
Randomly zeros 40% of the 1024 features during training.
Forces the model to not rely on any single feature — builds redundancy.
At test time, dropout is off and all features are used.

**Input:**  X-ray image [B, 1, 256, 256]
**Output:** Disease probabilities [B, 18]

---

## `src/classifier/train.py`

**What it does:** Trains CheXNet with binary cross-entropy loss.

**Why binary cross-entropy (not categorical)?**
```
Categorical CE:  one label per sample, softmax, sum to 1
Binary CE:       18 independent binary decisions, sigmoid, each 0 or 1

For each class independently:
  loss_i = -[y_i * log(p_i) + (1-y_i) * log(1-p_i)]
  total_loss = sum(loss_i for i in 18 classes)
```

**Runs twice:**
1. `--tag baseline` on `data/processed/` (real data only)
2. `--tag augmented` on `data/balanced/` (real + DDPM synthetic)

Comparing the two gives the ablation study evidence.

**Input:**  `XRayDataset(use_balanced=True/False)`
**Output:** `checkpoints/classifier/chexnet_baseline_best.pth`
            `checkpoints/classifier/chexnet_augmented_best.pth`

---

## `src/classifier/evaluate.py`

**What it does:** Measures classifier performance and generates the
ablation study comparison table.

**Two metrics:**

### AUC (Area Under ROC Curve)
Measures ranking quality regardless of threshold.
AUC=1.0 means the model always ranks positive cases higher than negative.
AUC=0.5 means random guessing.

### F1 Score
Harmonic mean of precision and recall.
F1=1.0 means perfect. F1=0.0 means useless.

**Per-class threshold finding:**
```python
for threshold in [0.05, 0.10, ..., 0.95]:
    y_pred = (probabilities > threshold).int()
    f1 = f1_score(y_true, y_pred)
    if f1 > best_f1:
        best_threshold = threshold
```

Default 0.5 misses rare classes (their max probability might be 0.15).
Sweeping thresholds finds the best operating point per class.

**Ablation table:**
When both baseline and augmented checkpoints exist, prints:
```
Class          Baseline F1   Augmented F1   Improvement
fibrosis       0.000         0.000          +0.000 →
emphysema      0.229         0.235          +0.007 ↑
...
```

**Input:**  Both checkpoint files + val set
**Output:** Printed results table + `outputs/classifier_eval/results_*.npy`

---

# PHASE 5 — BIOGPT REPORT GENERATION
> Goal: Generate radiology report text from X-ray image.

---

## `src/report_gen/projection.py`

**What it does:** Converts the VAE latent vector into 32 visual tokens
that BioGPT can read.

**Why this file exists:**
BioGPT works with token embeddings of dimension 768.
The VAE latent is [256, 16, 16] = 256 channels × 16×16 spatial grid.
We need to bridge these two very different formats.

**Architecture:**
```
z [B, 256, 16, 16]
  ↓ flatten spatial dimensions
[B, 256, 256]           (256 channels × 256 spatial positions)
  ↓ 1D Adaptive Average Pool → 32 positions
[B, 256, 32]
  ↓ transpose
[B, 32, 256]            (32 tokens, each 256-dim)
  ↓ Linear(256→768) + LayerNorm + GELU + Dropout(0.1) + Linear(768→768)
[B, 32, 768]            32 visual tokens in BioGPT's embedding space
```

**Why 1D pool not 2D pool?**
2D pool needs the output size to be a perfect spatial grid.
For 32 tokens: √32 = 5.65 → not an integer → 2D pool would silently
give 25 tokens (5×5) instead of 32. 1D pool works for any target size.
This was an actual bug that was fixed during training.

**Why GELU activation?**
GELU (Gaussian Error Linear Unit) is used in all transformer models
including BioGPT. Using it here keeps the projection's output distribution
compatible with what BioGPT expects.

**Input:**  VAE latent z [B, 256, 16, 16]
**Output:** Visual tokens [B, 32, 768]

**Connects to:** `model.py` calls this between the VAE encoder and BioGPT.

---

## `src/report_gen/model.py`

**What it does:** The full MedReportGenerator — combines VAE encoder +
VisualProjection + BioGPT into one model.

**This is the most important file in the project.**

**`__init__` — loading and freezing:**
```python
# 1. Load BioGPT (pretrained on 15M biomedical texts)
self.biogpt = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

# 2. Freeze ALL BioGPT layers first
for param in self.biogpt.parameters():
    param.requires_grad = False

# 3. Unfreeze only last 3 layers + output head
for layer in self.biogpt.biogpt.layers[-3:]:
    for param in layer.parameters():
        param.requires_grad = True

# 4. Load VAE encoder (already trained)
self.vae = VAE(...)
self.vae.load_state_dict(checkpoint["model_state_dict"])

# 5. Freeze VAE entirely
for param in self.vae.parameters():
    param.requires_grad = False

# 6. VisualProjection (trained from scratch)
self.projection = VisualProjection(...)
```

**`forward()` — training:**
```python
# Step 1: image → visual tokens
visual_tokens = self.encode_image(image)    # [B, 32, 768]

# Step 2: text tokens → embeddings (BioGPT's own embedding table)
text_embeds = self.biogpt.embed_tokens(input_ids)  # [B, 128, 768]

# Step 3: concatenate
combined = cat([visual_tokens, text_embeds], dim=1)  # [B, 160, 768]

# Step 4: attention mask — all ones
# WHY? Newer transformers versions compress output when mask has zeros
# This caused shape mismatches. All-ones mask is always safe.
mask = ones(B, 32 + 128)

# Step 5: labels — visual token positions get -100 (ignored in loss)
# We only compute loss on TEXT tokens, not image tokens
visual_labels = full((B, 32), -100)
combined_labels = cat([visual_labels, text_labels], dim=1)

# Step 6: forward through BioGPT
outputs = self.biogpt(inputs_embeds=combined, attention_mask=mask)

# Step 7: manual cross-entropy
# WHY MANUAL? Newer transformers has a bug: when labels are passed
# directly, it filters logits by non-(-100) count but leaves labels
# unfiltered → shape mismatch. Manual CE avoids this entirely.
loss = cross_entropy(shift_logits, shift_labels, ignore_index=-100)
```

**`generate()` — inference:**
```python
visual_tokens = self.encode_image(image)   # [1, 32, 768]

# BioGPT starts generating from the visual tokens
# It produces text word by word until <EOS> or max_length
generated = self.biogpt.generate(
    inputs_embeds=visual_tokens,
    max_new_tokens=128,
    num_beams=4,              # beam search: keep 4 candidate sequences
    repetition_penalty=1.3,   # discourage repeating phrases
    min_new_tokens=20,        # force at least 20 words
)
report = tokenizer.decode(generated[0])
```

**What is beam search?**
Instead of greedily picking the most likely next word, beam search
keeps the top-4 candidate sequences at each step and picks the one
with the highest overall probability at the end.
Greedy: fast but often repetitive or suboptimal
Beam search: slower but generates more coherent text

**Input:**  Image [B, 1, 256, 256] + tokenized report [B, 128] (training)
            Image [1, 1, 256, 256] (inference)
**Output:** Loss + logits (training) / report string (inference)

---

## `src/report_gen/train.py`

**What it does:** Fine-tunes MedReportGenerator on (image, report) pairs.

**Training objective:**
Given an image and the first N words of the report, predict word N+1.
This is standard language model training (next-token prediction).

```
Image + "FINDINGS: The heart is"
→ model should predict "enlarged"

Image + "FINDINGS: The heart is enlarged"
→ model should predict "."

...and so on
```

The visual tokens give the model image context for every word prediction.
When generating "enlarged," BioGPT can attend to the image token that
captured the heart region's size.

**Input:**  `XRayReportDataset` — paired (image, tokenized report)
**Output:** `checkpoints/report_gen/biogpt_best.pth`
            Contains: projection weights + last 3 BioGPT layer weights

---

## `src/report_gen/generate.py`

**What it does:** Inference script — runs the full generation on a
single image or the entire test set.

**`load_image_as_tensor()`:**
This helper is used by many other files (pipeline, demo).
Reads a PNG → grayscale → resize to 256×256 → normalize to [-1,1]
→ add batch and channel dims → [1, 1, 256, 256]

**Input:**  Single image path or test set
**Output:** Generated report string(s)

---

## `src/report_gen/evaluate.py`

**What it does:** Measures the quality of generated reports with
automatic metrics.

**Two metrics:**

### BLEU (Bilingual Evaluation Understudy)
Counts n-gram overlap between generated and reference report.
BLEU-4: what fraction of generated 4-word phrases appear in reference?
Range 0–1. Medical report BLEU scores are typically low (0.1–0.3) even
for good models because there are many valid ways to say the same thing.

### ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)
Measures longest common subsequence between generated and reference.
Captures word-order similarity, not just word presence.

**Important note:** These metrics are imperfect for medical text.
"Cardiomegaly is present" and "The heart is enlarged" mean the same
thing clinically but score 0% overlap. Human evaluation is ultimately
the gold standard.

---

## `src/report_gen/uncertainty.py`

**What it does:** Estimates how confident BioGPT is about a generated
report using Monte Carlo Dropout.

**The key insight — MC Dropout:**
```
Normal inference:
  model.eval()  ← dropout OFF
  same image → same report every time

MC Dropout:
  model.eval() then manually turn dropout layers back ON
  same image → slightly different report each time
  (different neurons zeroed each pass)

Run 20 times → 20 slightly different reports
Measure pairwise ROUGE-L similarity:
  All 20 say same thing → similarity = 0.95 → uncertainty = 0.05 → CONFIDENT
  20 different reports  → similarity = 0.40 → uncertainty = 0.60 → UNCERTAIN
```

**`enable_mc_dropout()`:**
```python
for module in model.modules():
    if isinstance(module, Dropout):
        module.train()   # force dropout layers back to training mode
```
This is careful — `model.eval()` is still called first (to disable
BatchNorm updates etc.). Only Dropout layers are selectively re-enabled.

**Why ROUGE-L not exact string match?**
"No acute cardiopulmonary disease" and "No acute cardiopulmonary abnormality"
are clinically identical but exact match gives 0% agreement, making a
confident model look uncertain. ROUGE-L gives ~0.85 for these — correct.

**Centroid report selection:**
Instead of returning the most frequent report (which fails when all 20
are slightly different), we pick the one with highest average ROUGE-L
similarity to all others — the "centroid" that best represents the
model's central tendency.

**Input:**  Trained model + single image
**Output:** Dict with best report, uncertainty score (0–1), status

---

# PHASE 6 — END-TO-END PIPELINE
> Goal: Combine all components and add a safety gate.

---

## `src/pipeline/consistency_check.py`

**What it does:** Checks whether the generated report text is consistent
with the CheXNet classifier's predictions.

**Why this exists:**
Language models hallucinate. BioGPT might output "no effusion" for an
image where CheXNet says effusion=0.82. In medicine, this contradiction
could be dangerous. This file catches it.

**How it works:**

Step 1: Extract what the report says about each pathology
```python
PATHOLOGY_KEYWORDS = {
    "effusion": ["effusion", "pleural fluid"],
    "cardiomegaly": ["cardiomegaly", "enlarged heart"],
    ...
}

# Scan report text for each pathology's keywords
report_flags = {p: any(kw in report.lower() for kw in kws)
                for p, kws in PATHOLOGY_KEYWORDS.items()}
```

Step 2: Compare with classifier probabilities
```python
for pathology in all_pathologies:
    report_says_present = report_flags[pathology]
    clf_says_present = prob[pathology] >= threshold[pathology]

    agree = (report_says_present == clf_says_present)
```

Step 3: Clinical mismatches trigger FLAG FOR REVIEW
Not all mismatches are equal. Missing "calcification" in the report is
less dangerous than missing "pneumothorax." Only clinical mismatches
(effusion, pneumothorax, cardiomegaly, pneumonia, mass, edema, etc.)
trigger the gate.

**Input:**  report text (string) + classifier_probs (dict)
**Output:** Dict with status (CONSISTENT/INCONSISTENT), mismatches list,
            clinical_mismatches list, per-pathology details

---

## `src/pipeline/run_pipeline.py`

**What it does:** The CLI script that runs all 4 stages end-to-end
on a single image.

**Stage sequence:**
```
Stage 1: Load image
         load_image_as_tensor(image_path) → [1, 1, 256, 256]

Stage 2: Generate report
         model.generate(image) → "FINDINGS: The heart is enlarged..."
         OR: mc_dropout_generate() → report + uncertainty score

Stage 3: Classify pathologies
         classifier(image) → {effusion: 0.72, cardiomegaly: 0.81, ...}

Stage 4: Consistency check
         check_consistency(report, probs) → CONSISTENT / INCONSISTENT

Stage 5: Gate decision
         INCONSISTENT OR high uncertainty → FLAG FOR REVIEW
         CONSISTENT AND low uncertainty  → AUTO-APPROVE
```

**Optional `--json_out` flag:**
Saves the full result as JSON — useful for batch evaluation or
integrating with a hospital information system.

**Input:**  Image path + checkpoint paths
**Output:** Report text + pathology probs + gate decision (printed + optional JSON)

---

# PHASE 7 — GRADIO DEMO UI
> Goal: Make the pipeline accessible to anyone without code.

---

## `src/demo/app.py`

**What it does:** Wraps the entire pipeline in a Gradio web interface.

**Architecture:**

### Model caching (`_cache` dict)
```python
_cache = {}

def _get_models():
    if "report" not in _cache:
        # Load all models ONCE
        _cache["report"] = load_report_model()
        _cache["classifier"] = load_classifier()
    return _cache["report"], _cache["classifier"]
```
Models are loaded on the FIRST click, then reused for all subsequent
requests. Loading takes ~30 seconds. Without caching, every click would
reload — unusable.

### `_prob_chart()` — matplotlib figure
Creates a horizontal bar chart of all 18 pathology probabilities.
Bars are red if probability ≥ 0.5 (positive), blue if below.
Sorted ascending so highest probability appears at the top.

### `_consistency_html()` — HTML table
Builds an HTML table showing for each pathology:
- What the report says (yes/no)
- What the classifier predicts (POS/NEG + exact probability)
- Whether they agree (✅/❌)
Rows with disagreements are highlighted in light red.

### `_gate_html()` — colored badge
Returns an HTML div — green for AUTO-APPROVE, red for FLAG FOR REVIEW.

### `run_inference()` — main function
Called by Gradio on every button click:
```python
def run_inference(image_path, use_uncertainty, mc_passes):
    gen, clf, device = _get_models()
    image = load_image_as_tensor(image_path).to(device)

    report = gen.generate(image)           # BioGPT
    probs  = clf(image)                    # CheXNet
    consistency = check_consistency(...)   # gate
    gate = "AUTO-APPROVE" or "FLAG FOR REVIEW"

    return report_text, prob_figure, consistency_html, gate_html
```

### `build_demo()` — Gradio layout
```
Row 1:
  Left column:   Image upload + Options accordion (uncertainty toggle)
                 Generate Report button
  Right column:  Report textbox + Gate badge

Row 2:
  Left:   Probability bar chart (gr.Plot)
  Right:  Consistency HTML table
```

### PWA (Progressive Web App)
Gradio 6.x automatically serves a web manifest. Any browser
visiting the URL sees an "Install App" prompt. Clicking it adds
an icon to the home screen — works offline, opens like a native app.
Zero extra code needed.

**Input:**  Uploaded image file (Gradio passes as temp file path)
**Output:** Report text, matplotlib figure, consistency HTML, gate HTML

---

## `spaces_app.py` (project root)

**What it does:** Entry point for HuggingFace Spaces deployment.

**Why separate from `app.py`:**
On HuggingFace Spaces, checkpoints can't be committed to git (too large).
This file downloads them from a HF model repository at startup:
```python
for filename in ["vae_best.pth", "biogpt_best.pth", "chexnet_augmented_best.pth"]:
    hf_hub_download(repo_id=HF_MODEL_REPO, filename=filename)
```
Then sets environment variables pointing to the downloaded paths,
and launches the same `build_demo()` from `app.py`.

---

# HOW EVERYTHING CONNECTS — DATA FLOW SUMMARY

```
download.py
    │ raw images + XML reports
    ▼
parse_reports.py + preprocess.py
    │ data/processed/labels.csv + data/processed/images/
    ▼
dataset.py (XRayDataset)
    │ batches of (image tensor, label, multi_hot)
    ├──────────────────────────────────────────────────┐
    ▼                                                  ▼
ddpm/train.py                                    vae/train.py
    │ checkpoints/ddpm/ddpm_best.pth                   │ checkpoints/vae/vae_best.pth
    ▼                                                  │
ddpm/sample.py                                         │
    │ data/balanced/ (real + synthetic)                │
    ▼                                                  │
classifier/train.py ←──────────────────────────────────┘ (uses balanced data)
    │ checkpoints/classifier/chexnet_*.pth
    │
    └──────────────────────────────────────────────────┐
                                                       │
dataset.py (XRayReportDataset)                         │
    │ batches of (image, tokenized report)             │
    ▼                                                  │
report_gen/train.py                                    │
    │ checkpoints/report_gen/biogpt_best.pth            │
    │                                                  │
    └──────────────────┬───────────────────────────────┘
                       │
                       ▼
            pipeline/run_pipeline.py
            demo/app.py
                loads all 3 checkpoints
                image → report → probs → gate → output
```

---

# QUICK REFERENCE — What to say for each file in an interview

| File | One-line purpose |
|------|-----------------|
| `dataset.py` | Central data loader — image + label + report for every training script |
| `diffusion.py` | DDPM math — add noise forward, remove noise backward |
| `conditioning.py` | Convert timestep + disease class into embedding vectors for UNet |
| `unet.py` | Neural network that predicts noise at each denoising step |
| `sample.py` | Use trained DDPM to generate synthetic X-rays per class |
| `encoder.py` | CNN that compresses X-ray to latent vector [256,16,16] |
| `vae.py` | Combines encoder + decoder, computes reconstruction + KL loss |
| `chexnet.py` | DenseNet121 with 18-class sigmoid head for multi-label classification |
| `projection.py` | Convert VAE latent to 32 visual tokens BioGPT can read |
| `model.py` | Full pipeline: VAE encoder → projection → BioGPT → report text |
| `train.py` (report_gen) | Fine-tune BioGPT on (image, report) pairs |
| `uncertainty.py` | MC Dropout: run 20 passes, measure ROUGE-L variance → uncertainty score |
| `consistency_check.py` | Compare report keywords vs classifier predictions → flag mismatches |
| `run_pipeline.py` | CLI end-to-end: image → report → probs → gate decision |
| `app.py` | Gradio web UI wrapping the full pipeline |
