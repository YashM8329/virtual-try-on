"""
inpainting.py
Phase 2 & 3 — Prompt Engineering & ControlNet Inpainting Inference.

Full inference pipeline:
  1. Prepare PIL image + soft mask for diffusion
  2. Generate Canny control image for body-conformant generation
  3. Run StableDiffusionControlNetInpaintPipeline
  4. Alpha-blend result back onto original using soft mask
"""

import torch
import numpy as np
from PIL import Image

from image_utils import make_canny_control_image
from model_loader import DEVICE_DIFF

# ── Prompt Configuration ───────────────────────────────────────────────────────
POSITIVE_PROMPT = (
    "(bright neon lime green zip-up hoodie:1.9), "
    "(vivid solid lime green:1.7), "
    "(black spray paint graffiti all over the hoodie:1.5), "
    "black cartoon doodles, smiley face graffiti tags, "
    "cotton fleece hoodie material, hood visible at top, "
    "front zipper, kangaroo pocket, "
    "photorealistic clothing, sharp detail, 8k"
)

NEGATIVE_PROMPT = (
    "(knitted:1.8), (sweater:1.8), (wool:1.7), "
    "(vest:1.6), (inner layer:1.5), "
    "(black hoodie:1.8), (dark:1.7), (denim:1.6), "
    "skin on clothes, neck exposed in garment, "
    "deformed, blurry, unrealistic, low quality"
)

# ── Diffusion Hyperparameters ──────────────────────────────────────────────────
DIFFUSION_CONFIG = {
    "num_inference_steps": 50,
    "guidance_scale": 12.0,   # Higher = stronger prompt adherence
    "strength": 0.97,          # 0.97 = preserves body lighting/shadows
    "seed": 42,                # Fixed seed for texture consistency
}


def run_inpainting(img_pil: Image.Image, soft_mask: np.ndarray, pipe, config: dict = None) -> Image.Image:
    """
    Run ControlNet inpainting on a single image.

    Args:
        img_pil   : PIL Image (RGB) — preprocessed input
        soft_mask : np.ndarray (H, W) uint8 — Gaussian-blurred clothing mask
        pipe      : loaded StableDiffusionControlNetInpaintPipeline
        config    : diffusion hyperparameter dict (defaults to DIFFUSION_CONFIG)

    Returns:
        PIL Image — alpha-composited result (generated garment blended onto original)
    """
    if config is None:
        config = DIFFUSION_CONFIG

    orig_size = img_pil.size   # (W, H)
    h, w = img_pil.size[1], img_pil.size[0]

    generator = torch.Generator(device=DEVICE_DIFF).manual_seed(config["seed"])

    mask_pil = Image.fromarray(soft_mask).convert("L")
    control_image = make_canny_control_image(img_pil, low=80, high=180)

    result = pipe(
        prompt=POSITIVE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=img_pil,
        mask_image=mask_pil,
        control_image=control_image,
        num_inference_steps=config["num_inference_steps"],
        guidance_scale=config["guidance_scale"],
        strength=config["strength"],
        controlnet_conditioning_scale=0.2,   # lower = more natural
        generator=generator,
        height=h,
        width=w,
    ).images[0]

    # Force resize back — SD sometimes changes dimensions slightly
    result = result.resize(orig_size, Image.LANCZOS)

    # Alpha-blend generated result onto original using soft mask
    orig_np = np.array(img_pil).astype(np.float32)
    result_np = np.array(result).astype(np.float32)
    alpha = (soft_mask.astype(np.float32) / 255.0)[..., np.newaxis]
    composited_np = (result_np * alpha + orig_np * (1 - alpha)).astype(np.uint8)

    return Image.fromarray(composited_np)
