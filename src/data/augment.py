"""
augment.py
----------
Image augmentations for chest X-ray training.

Why augmentation for medical images:
  - X-rays have limited variation (patient positioning, equipment)
  - Augmentation prevents overfitting on small datasets
  - MUST be conservative — don't distort diagnostic content

Safe augmentations for X-rays:
  ✅ Horizontal flip   (patient orientation)
  ✅ Small rotation    (±10°, patient not perfectly aligned)
  ✅ Brightness/gamma  (different X-ray equipment)
  ✅ Gaussian noise    (sensor noise simulation)
  ✅ Random crop       (slight framing differences)

Unsafe augmentations for X-rays (DO NOT USE):
  ❌ Vertical flip     (flips anatomical structures)
  ❌ Large rotation    (destroys spatial meaning)
  ❌ Color jitter      (X-rays are grayscale, color = artifact)
  ❌ Elastic deform    (distorts lung shape = wrong diagnosis)

Usage:
  from src.data.augment import get_train_transforms, get_val_transforms
"""

import cv2
import numpy as np
import torch
from typing import Callable

# ------------------------------------------------------------------ #
#  INDIVIDUAL AUGMENTATIONS                                          #
# ------------------------------------------------------------------ #

def random_horizontal_flip(img: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Flips image horizontally with probability p.
    Safe for X-rays — left/right chest are anatomically similar enough.
    """
    if np.random.random() < p:
        return cv2.flip(img, 1)
    return img


def random_rotation(img: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """
    Rotates image by a random angle within [-max_angle, +max_angle].
    Simulates patient not perfectly aligned on the imaging table.
    """
    angle = np.random.uniform(-max_angle, max_angle)
    h, w  = img.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Apply rotation
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags       = cv2.INTER_LANCZOS4,
        borderMode  = cv2.BORDER_REFLECT,  # fill edges with reflection
    )
    return rotated


def random_brightness(img: np.ndarray, factor_range: tuple = (0.85, 1.15)) -> np.ndarray:
    """
    Adjusts image brightness by a random factor.
    Simulates different X-ray exposure settings.
    """
    factor = np.random.uniform(*factor_range)
    img_float = img.astype(np.float32) * factor
    img_clipped = np.clip(img_float, 0, 255)
    return img_clipped.astype(np.uint8)


def random_gamma(img: np.ndarray, gamma_range: tuple = (0.8, 1.2)) -> np.ndarray:
    """
    Applies random gamma correction.
    Simulates different display/scanner gamma settings.
    """
    gamma = np.random.uniform(*gamma_range)
    inv_gamma = 1.0 / gamma

    # Build lookup table (faster than pixel-by-pixel)
    table = np.array([
        (i / 255.0) ** inv_gamma * 255
        for i in range(256)
    ]).astype(np.uint8)

    return cv2.LUT(img, table)


def random_gaussian_noise(img: np.ndarray, std_range: tuple = (0, 10)) -> np.ndarray:
    """
    Adds random Gaussian noise.
    Simulates sensor noise in different X-ray equipment.
    """
    std = np.random.uniform(*std_range)
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def random_crop_and_resize(
    img       : np.ndarray,
    crop_scale: tuple = (0.85, 1.0),
    size      : int   = 256,
) -> np.ndarray:
    """
    Randomly crops a portion of the image then resizes back to `size`.
    Simulates slight framing differences between scans.
    """
    h, w  = img.shape[:2]
    scale = np.random.uniform(*crop_scale)

    new_h = int(h * scale)
    new_w = int(w * scale)

    # Random top-left corner for the crop
    top  = np.random.randint(0, h - new_h + 1)
    left = np.random.randint(0, w - new_w + 1)

    cropped = img[top:top + new_h, left:left + new_w]
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return resized


# ------------------------------------------------------------------ #
#  TRANSFORM PIPELINES                                               #
# ------------------------------------------------------------------ #

def to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Converts numpy [H, W] grayscale image → torch [1, H, W] tensor in [0, 1].
    This is always the last step in any transform pipeline.
    """
    tensor = torch.from_numpy(img).float() / 255.0
    return tensor.unsqueeze(0)   # add channel dimension


def get_train_transforms(image_size: int = 256) -> Callable:
    """
    Returns the training augmentation pipeline.
    Each augmentation is applied with some probability to keep
    the dataset diverse without destroying every original image.
    """
    def transform(img: np.ndarray) -> torch.Tensor:
        # Apply augmentations in sequence
        img = random_horizontal_flip(img, p=0.5)
        img = random_rotation(img, max_angle=8.0)
        img = random_brightness(img, factor_range=(0.85, 1.15))
        img = random_gamma(img, gamma_range=(0.85, 1.15))
        img = random_gaussian_noise(img, std_range=(0, 8))
        img = random_crop_and_resize(img, crop_scale=(0.90, 1.0), size=image_size)

        # Always last: convert to tensor
        return to_tensor(img)

    return transform


def get_val_transforms(image_size: int = 256) -> Callable:
    """
    Returns the validation/test transform pipeline.
    No augmentation — only resize and tensor conversion.
    Deterministic: same image → same tensor every time.
    """
    def transform(img: np.ndarray) -> torch.Tensor:
        # Just resize to ensure correct size
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)
        return to_tensor(img)

    return transform


# ------------------------------------------------------------------ #
#  VERIFICATION — run this file to see augmentations visually        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    img_dir = Path("data/processed/images")
    imgs    = list(img_dir.glob("*.png"))

    if not imgs:
        print("[ERROR] No processed images found. Run preprocess.py first.")
        exit()

    # Load one sample image
    original = cv2.imread(str(imgs[0]), cv2.IMREAD_GRAYSCALE)
    print(f"Testing augmentations on: {imgs[0].name}")
    print(f"Original shape: {original.shape}")

    # Apply each augmentation and visualize
    augmentations = {
        "Original"           : original,
        "Horizontal Flip"    : random_horizontal_flip(original, p=1.0),
        "Rotation (+8°)"     : random_rotation(original, max_angle=8.0),
        "Brightness ×1.15"   : random_brightness(original, (1.15, 1.15)),
        "Gamma 0.8"          : random_gamma(original, (0.8, 0.8)),
        "Gaussian Noise"     : random_gaussian_noise(original, (5, 5)),
        "Random Crop"        : random_crop_and_resize(original, (0.85, 0.85)),
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (name, aug_img) in enumerate(augmentations.items()):
        axes[i].imshow(aug_img, cmap="gray")
        axes[i].set_title(name, fontsize=10)
        axes[i].axis("off")

    axes[-1].axis("off")  # hide last empty subplot

    Path("outputs").mkdir(exist_ok=True)
    plt.suptitle("X-Ray Augmentation Verification", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/augmentation_verification.png", dpi=100)
    print("Saved → outputs/augmentation_verification.png")