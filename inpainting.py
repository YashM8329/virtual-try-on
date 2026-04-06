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
    "a high-quality (lime green hoodie:1.9), "
    "vibrant solid color (hex code #5DDF2E:1.8), "
    "(dense black graffiti and urban street-art pattern:1.7) covering the entire hoodie, "
    "hand-drawn black (smiley face doodles:1.5), spray paint tags, stylized black text graffiti, "
    "heavy cotton fleece fabric texture, black drawstrings visible, "
    "hood visible at the top, "
    "photorealistic clothing, sharp crisp details, cinematic lighting, 8k"
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
    "num_inference_steps": 25,    # ← bumped 20→25 to handle 768px detail
    "guidance_scale": 7.5,        # ← optimal for DPM++
    "strength": 0.97,
    "diffusion_size": 768,        # ← was 512; 768 gives finer latent grid (96×96 vs 64×64)
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
) -> Image.Image:
    """
    Run MultiControlNet (OpenPose + Canny) inpainting on a single image.

    Pipeline
    ────────
    1. Downscale image + mask to diffusion_size (512 px longest edge).
    2. Generate OpenPose skeleton image and Canny edge map at 512 px.
    3. Run SD inpainting with 20 DPM++ steps.
    4. Upscale result back to original resolution.
    5. Alpha-blend with original using the original-resolution soft mask.

    Args:
        img_pil      : PIL Image (RGB) — already cropped + enhanced
        soft_mask    : np.ndarray (H, W) uint8 — Gaussian-blurred clothing mask
        pipe         : loaded StableDiffusionControlNetInpaintPipeline
        config       : override dict (defaults to DIFFUSION_CONFIG)
        openpose_gen : OpenposeDetector instance (from controlnet-aux)
        image_path   : original file path used to derive a per-image seed

    Returns:
        PIL Image — garment composited onto original at original resolution
    """
    if config is None:
        config = DIFFUSION_CONFIG

    orig_size = img_pil.size          # (W, H) — save for final upscale
    orig_h, orig_w = orig_size[1], orig_size[0]

    # ── 1. Downscale to diffusion_size for faster UNet passes ─────────────────
    diff_size = config.get("diffusion_size", 512)
    scale     = diff_size / max(orig_h, orig_w)
    diff_w    = int(round(orig_w * scale / 8)) * 8
    diff_h    = int(round(orig_h * scale / 8)) * 8

    img_small  = img_pil.resize((diff_w, diff_h), Image.LANCZOS)
    mask_small = Image.fromarray(soft_mask).convert("L").resize(
        (diff_w, diff_h), Image.LANCZOS
    )

    # ── 2. Control images at diffusion resolution ──────────────────────────────
    canny_image = make_canny_control_image(img_small, low=80, high=180)

    if openpose_gen is not None:
        pose_image = openpose_gen(img_small)
    else:
        # Fallback: reuse Canny when OpenPose is unavailable
        pose_image = canny_image

    control_images = [pose_image, canny_image]

    # ── 3. Per-image deterministic seed ───────────────────────────────────────
    seed      = _make_seed(image_path) if image_path else 42
    # NOTE: use "cpu" for the generator when enable_model_cpu_offload is active
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # ── 4. Diffusion at 512 px ─────────────────────────────────────────────────
    result_small = pipe(
        prompt=POSITIVE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=img_small,
        mask_image=mask_small,
        control_image=control_images,
        num_inference_steps=config["num_inference_steps"],
        guidance_scale=config["guidance_scale"],
        strength=config["strength"],
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