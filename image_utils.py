"""
image_utils.py
Shared image helpers used across the pipeline.

  preprocess_image     — Pipeline step 1. Loads an image from disk and resizes
                         it to 1024 px (longest edge) snapped to a multiple of 8,
                         which is required by the Stable Diffusion UNet.

  make_canny_control_image — Called inside run_inpainting (step 6) to produce
                         the Canny edge map fed to the Canny ControlNet.

  save_images          — Debug utility: writes multiple images side-by-side to
                         a single PNG file (not called during normal pipeline runs).
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts / headless servers
import matplotlib.pyplot as plt


def preprocess_image(image_input, target_size=1024):
    """
    Load and resize an image so its longest edge equals `target_size` and both
    dimensions are multiples of 8 (required by the SD UNet).

    Args:
        image_input : file path (str) or PIL.Image
        target_size : longest-edge target in pixels (default 1024)

    Returns:
        img_pil  : PIL.Image (RGB) — passed to run_inpainting and FaceEnhancer
        img_bgr  : np.ndarray (H, W, 3) uint8 BGR — passed to remove_background
        (new_h, new_w) : resized dimensions
    """
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
    else:
        img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_h = int(round(h * scale / 8)) * 8
    new_w = int(round(w * scale / 8)) * 8
    img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return img_pil, img_bgr, (new_h, new_w)


def make_canny_control_image(img_pil, low=100, high=200):
    """
    Generate a Canny edge map for the Canny ControlNet (pipeline step 6).

    The edge map is computed on the downscaled diffusion-resolution image
    passed in from run_inpainting, so body edges guide garment fit at the
    exact resolution the UNet operates on.

    Args:
        img_pil : PIL.Image (RGB) at diffusion resolution
        low     : lower hysteresis threshold for cv2.Canny
        high    : upper hysteresis threshold for cv2.Canny

    Returns:
        PIL.Image (RGB) edge map — white edges on black background
    """
    img_gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, low, high)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def save_images(images: dict, save_path: str, figsize=(18, 6)):
    """
    Debug utility — save multiple images side-by-side with titles to a single PNG.

    Args:
        images    : dict of {title: PIL.Image or np.ndarray}
        save_path : output path (e.g. "masks/debug_foo.png")
        figsize   : matplotlib figure size in inches
    """
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, images.items()):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
