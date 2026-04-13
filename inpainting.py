"""
inpainting.py
Pipeline step 6 — MultiControlNet inpainting with IP-Adapter.

Replaces the masked clothing region with the neon lime green hoodie using
Stable Diffusion 1.5 conditioned on:
  - OpenPose control image  (guides garment fit to body pose)
  - Canny control image     (preserves body edges and silhouette)
  - IP-Adapter reference    (garment.jpg used for colour/texture reference)
  - Text prompt             (hoodie description + quality boosters)

The diffusion pass runs at a reduced resolution (`diffusion_size`, default 768 px)
for VRAM efficiency, then the result is upscaled back to the original 1024 px and
alpha-composited onto the white-background image using the soft clothing mask.
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
# These are the defaults; main.py passes its own config dict to run_inpainting()
# which is merged on top (main.py values take precedence).
DIFFUSION_CONFIG = {
    "num_inference_steps": 20,  # DPM++ 2M Karras reaches high quality in 15-20 steps
    "guidance_scale": 10.0,      # CFG scale — higher = stricter prompt adherence
    "strength": 0.99,           # Inpainting strength — near 1.0 fully replaces the masked area
    "diffusion_size": 1024,      # Inference resolution; main.py overrides this to 1024
}

# ── ControlNet conditioning scales for [OpenPose, Canny] ──────────────────────
# OpenPose (0.7): strong pose guidance so the garment fits the body shape.
# Canny (0.4): lighter edge guidance to preserve silhouette without over-constraining.
CONTROLNET_SCALES = [0.7, 0.4]


def _make_seed(image_path: str) -> int:
    """Deterministic per-image seed derived from the filename (MD5 hash).
    Keeps results reproducible across runs without cross-image seed bleed."""
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
    Run MultiControlNet SD 1.5 inpainting conditioned on pose, edges, and garment reference.

    Args:
        img_pil          : white-background PIL image at 1024 px (output of remove_background).
        soft_mask        : (H, W) uint8 — Gaussian-blurred clothing mask from apply_neck_blend().
        pipe             : StableDiffusionControlNetInpaintPipeline loaded in model_loader.py.
        config           : optional dict of hyperparameter overrides (merged over DIFFUSION_CONFIG).
        openpose_gen     : OpenposeDetector loaded in model_loader.py.
        image_path       : source file path used to derive a deterministic seed.
        garment_image_pil: garment.jpg as PIL image for IP-Adapter reference conditioning.
                           If None, the pipeline runs in text-prompt-only mode.

    Returns:
        PIL.Image (RGB) at the original 1024 px resolution — the inpainted result
        alpha-composited back onto the white-background image.
    """
    # Merge caller overrides on top of module defaults
    full_config = DIFFUSION_CONFIG.copy()
    if config:
        full_config.update(config)

    orig_size = img_pil.size
    orig_h, orig_w = orig_size[1], orig_size[0]

    # ── 1. Downscale to diffusion_size ────────────────────────────────────────
    # Running the UNet at a reduced resolution cuts VRAM and step time;
    # the result is upscaled back to 1024 px after generation.
    diff_size = full_config.get("diffusion_size", 768)
    scale     = diff_size / max(orig_h, orig_w)
    diff_w    = int(round(orig_w * scale / 8)) * 8
    diff_h    = int(round(orig_h * scale / 8)) * 8

    img_small  = img_pil.resize((diff_w, diff_h), Image.LANCZOS)
    mask_small = Image.fromarray(soft_mask).convert("L").resize(
        (diff_w, diff_h), Image.LANCZOS
    )

    # ── 2. Build ControlNet conditioning images at diffusion resolution ────────
    canny_image = make_canny_control_image(img_small, low=80, high=180)
    pose_image  = openpose_gen(img_small) if openpose_gen is not None else canny_image
    control_images = [pose_image, canny_image]  # order matches CONTROLNET_SCALES

    # ── 3. Deterministic per-image seed ───────────────────────────────────────
    seed      = _make_seed(image_path) if image_path else 42
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # ── 4. SD 1.5 inpainting pass ─────────────────────────────────────────────
    result_small = pipe(
        prompt=POSITIVE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=img_small,
        mask_image=mask_small,
        control_image=control_images,
        ip_adapter_image=garment_image_pil,
        num_inference_steps=full_config["num_inference_steps"],
        guidance_scale=full_config["guidance_scale"],
        strength=full_config["strength"],
        controlnet_conditioning_scale=CONTROLNET_SCALES,
        generator=generator,
        height=diff_h,
        width=diff_w,
    ).images[0]

    # ── 5. Upscale result back to original 1024 px ────────────────────────────
    result = result_small.resize(orig_size, Image.LANCZOS)

    # ── 6. Alpha-composite generated garment onto white-background image ───────
    # Uses the original-resolution soft mask so blending matches the mask
    # that was given to the UNet (not the downscaled version).
    orig_np    = np.array(img_pil).astype(np.float32)
    result_np  = np.array(result).astype(np.float32)
    alpha      = (soft_mask.astype(np.float32) / 255.0)[..., np.newaxis]
    composited = (result_np * alpha + orig_np * (1 - alpha)).astype(np.uint8)

    return Image.fromarray(composited)