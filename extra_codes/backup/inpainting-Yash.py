"""
inpainting.py
Phase 2 & 3 — Prompt Engineering & ControlNet Inpainting Inference.
Optimized for 6 GB VRAM (RTX 3050) — uses CPU offload pipeline.
"""

import torch
import numpy as np
from PIL import Image

from image_utils import make_canny_control_image

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE_DIFF = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    "guidance_scale": 12.0,
    "strength": 0.97,
    "seed": 42,
}


def run_inpainting(img_pil: Image.Image, soft_mask: np.ndarray, pipe, config: dict = None) -> Image.Image:
    """
    Run ControlNet inpainting. Pipeline uses CPU offload so we do NOT
    manually move tensors to DEVICE_DIFF — the pipeline handles placement.
    """
    if config is None:
        config = DIFFUSION_CONFIG

    orig_size = img_pil.size       # (W, H)
    h, w = img_pil.size[1], img_pil.size[0]

    # Generator must be on CPU when using enable_model_cpu_offload
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])

    mask_pil      = Image.fromarray(soft_mask).convert("L")
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
        controlnet_conditioning_scale=0.2,
        generator=generator,
        height=h,
        width=w,
    ).images[0]

    # Force resize back — SD sometimes changes dimensions slightly
    result = result.resize(orig_size, Image.LANCZOS)

    # Alpha-blend generated result onto original using soft mask
    orig_np   = np.array(img_pil).astype(np.float32)
    result_np = np.array(result).astype(np.float32)
    alpha     = (soft_mask.astype(np.float32) / 255.0)[..., np.newaxis]
    composited_np = (result_np * alpha + orig_np * (1 - alpha)).astype(np.uint8)

    return Image.fromarray(composited_np)