"""
inpainting.py
Phase 2 & 3 — Prompt Engineering & ControlNet Inpainting Inference.

Key changes vs previous version
────────────────────────────────
1. num_inference_steps: 50 → 20
   DPMSolverMultistepScheduler (loaded in model_loader.py) reaches
   equivalent quality in 20 steps.  ~60 % time saving.

2. Diffusion at 512 px, upscale result to original size.
   UNet attention maps shrink from 96×96 → 64×64.
   ~2.25× faster per step; also reduces peak VRAM.

3. Per-image deterministic seed (MD5 hash of filename).
   Removes cross-image seed bleed while staying reproducible.

4. controlnet_conditioning_scale raised: [0.7, 0.4]
   (was 0.2 on single ControlNet) — OpenPose guides garment fit,
   Canny preserves body edges.

5. guidance_scale lowered: 12.0 → 7.5
   High CFG costs extra UNet forward passes per step on some
   schedulers and adds harshness without quality gain on DPM++.
"""

import hashlib
import torch
import numpy as np
from PIL import Image

from image_utils import make_canny_control_image
from model_loader import DEVICE_DIFF

# ── Prompt Configuration ───────────────────────────────────────────────────────
POSITIVE_PROMPT = (
    "(neon lime green #5DDF2E hoodie:1.9), "
    "(dense black rectangle-shaped doodles and graffiti motifs:1.7), "
    "(hand-drawn black ink street-art sketches:1.5), "
    "cotton fleece, black drawstrings, kangaroo pocket, "
    "photorealistic, sharp fabric texture, 8k, cinematic lighting"
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
    "num_inference_steps": 20,     # ← Increased for high quality
    "guidance_scale": 8.0,        # ← higher CFG for better prompt adherence
    "strength": 0.99,             # ← ensures garment fully replaces area
    "diffusion_size": 768,        # ← 768px fixes blurriness
}

# ── Conditioning scales for [OpenPose, Canny] ─────────────────────────────────
CONTROLNET_SCALES = [0.7, 0.4]


def _make_seed(image_path: str) -> int:
    """Deterministic per-image seed from filename. Reproducible, no cross-bleed."""
    return int(hashlib.md5(image_path.encode()).hexdigest()[:8], 16) % (2 ** 32)


def run_inpainting(
    img_pil: Image.Image,
    soft_mask: np.ndarray,
    pipe,
    config: dict = None,
    openpose_gen=None,
    image_path: str = "",
    garment_image_pil: Image.Image = None,
) -> Image.Image:
    """
    Run MultiControlNet (OpenPose + Canny) inpainting with IP-Adapter (image reference).
    """
    # Merge provided config with defaults
    full_config = DIFFUSION_CONFIG.copy()
    if config:
        full_config.update(config)

    orig_size = img_pil.size
    orig_h, orig_w = orig_size[1], orig_size[0]

    # ── 1. Downscale to diffusion_size for faster UNet passes ─────────────────
    diff_size = full_config.get("diffusion_size", 768) # Default to 768 for higher quality
    scale     = diff_size / max(orig_h, orig_w)
    diff_w    = int(round(orig_w * scale / 8)) * 8
    diff_h    = int(round(orig_h * scale / 8)) * 8

    img_small  = img_pil.resize((diff_w, diff_h), Image.LANCZOS)
    mask_small = Image.fromarray(soft_mask).convert("L").resize(
        (diff_w, diff_h), Image.LANCZOS
    )

    # ── 2. Control images ──────────────────────────────────────────────────────
    canny_image = make_canny_control_image(img_small, low=80, high=180)
    if openpose_gen is not None:
        pose_image = openpose_gen(img_small)
    else:
        pose_image = canny_image
    control_images = [pose_image, canny_image]

    # ── 3. Per-image seed ─────────────────────────────────────────────────────
    seed      = _make_seed(image_path) if image_path else 42
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # ── 4. Diffusion with IP-Adapter ──────────────────────────────────────────
    result_small = pipe(
        prompt=POSITIVE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=img_small,
        mask_image=mask_small,
        control_image=control_images,
        ip_adapter_image=garment_image_pil, # Use garment image for reference
        num_inference_steps=full_config["num_inference_steps"],
        guidance_scale=full_config["guidance_scale"],
        strength=full_config["strength"],
        controlnet_conditioning_scale=CONTROLNET_SCALES,
        generator=generator,
        height=diff_h,
        width=diff_w,
    ).images[0]

    # ── 5. Upscale result back to original resolution ─────────────────────────
    result = result_small.resize(orig_size, Image.LANCZOS)

    # ── 6. Alpha-blend onto original using original-resolution soft mask ───────
    orig_np   = np.array(img_pil).astype(np.float32)
    result_np = np.array(result).astype(np.float32)
    alpha     = (soft_mask.astype(np.float32) / 255.0)[..., np.newaxis]
    composited = (result_np * alpha + orig_np * (1 - alpha)).astype(np.uint8)

    return Image.fromarray(composited)