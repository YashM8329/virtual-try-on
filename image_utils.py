"""
image_utils.py
Helper utilities:
- Image preprocessing (resize to SD-compatible multiple of 8)
- Canny edge map generation for ControlNet
- Visualization helpers (save side-by-side debug images)
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for VS Code / scripts
import matplotlib.pyplot as plt


def preprocess_image(image_input, target_size=768):
    """
    Load image and resize to nearest multiple of 8 for SD stability.

    Args:
        image_input: file path (str) or PIL.Image
        target_size: longest edge target in pixels

    Returns:
        (PIL.Image RGB, np.ndarray BGR, (new_h, new_w))
    """
    if isinstance(image_input, str):
        img_bgr = cv2.imread(image_input)
    else:
        img_bgr = np.array(image_input)
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

    h, w = img_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_h = int(round(h * scale / 8)) * 8
    new_w = int(round(w * scale / 8)) * 8
    img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    return img_pil, img_bgr, (new_h, new_w)


def make_canny_control_image(img_pil, low=100, high=200):
    """Generate Canny edge map from PIL image for ControlNet conditioning."""
    img_gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, low, high)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def save_images(images: dict, save_path: str, figsize=(18, 6)):
    """
    Save multiple PIL images side-by-side with titles to disk.

    Args:
        images: dict of {title: PIL.Image or np.ndarray}
        save_path: full path where the figure is saved (e.g. masks/debug_foo.png)
        figsize: matplotlib figure size
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
