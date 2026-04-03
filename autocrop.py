"""
autocrop.py
Pose-Aware Auto-Crop — Crops the input image HORIZONTALLY ONLY,
trimming the width so the frame spans from the leftmost to the
rightmost hand/wrist landmark. Full image height is preserved.

Width is snapped to the nearest lower multiple of 8 (SD requirement).
No margin is added — the crop is tight to the outermost hand x-coords.

Usage (drop-in, no core logic change):
    from autocrop import crop_to_hands
    img_pil, img_bgr, (h, w) = preprocess_image(image_path)
    img_pil, img_bgr, (h, w) = crop_to_hands(img_pil, img_bgr, lm_px)
"""

import os
import cv2
import numpy as np
from PIL import Image

CROP_DIR = "cropped"
os.makedirs(CROP_DIR, exist_ok=True)


# MediaPipe landmark indices used for horizontal boundary
_WRIST_IDXS = [15, 16]               # Left + right wrists
_HAND_IDXS  = [17, 18, 19, 20, 21, 22]  # Pinky/index/thumb tips


def crop_to_hands(
    img_pil: Image.Image,
    img_bgr: np.ndarray,
    lm_px: dict,
    fname: str = None,
):
    """
    Crop the image horizontally so the frame spans tightly from the
    leftmost to the rightmost hand/wrist landmark. Height unchanged.

    Args:
        img_pil  : PIL Image (RGB) — must match img_bgr spatially
        img_bgr  : np.ndarray (H, W, 3) BGR
        lm_px    : dict {idx: (x_px, y_px)} from extract_pose_landmarks()

    Returns:
        (cropped_pil, cropped_bgr, (new_h, new_w))
        — same tuple shape as preprocess_image(), slots in directly.
    """
    h, w = img_bgr.shape[:2]

    # ── Collect all hand/wrist x-coordinates ───────────────────────────────
    x_candidates = []
    for idx in _WRIST_IDXS + _HAND_IDXS:
        if idx in lm_px:
            x_candidates.append(lm_px[idx][0])

    if x_candidates:
        margin_px = int(w * 0.10)                        # 10% of original width
        left_x  = max(0, min(x_candidates) - margin_px)
        right_x = min(w, max(x_candidates) + margin_px)
    else:
        # Fallback: keep full width
        left_x, right_x = 0, w

    # ── Snap right boundary down to multiple of 8 (SD requirement) ─────────
    crop_w  = right_x - left_x
    snap_w  = (crop_w // 8) * 8        # floor: at most 7 px trimmed from right
    right_x = left_x + snap_w          # recalculate the right edge

    # ── Horizontal-only crop; full height kept ──────────────────────────────
    cropped_bgr = img_bgr[:, left_x:right_x, :]
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    cropped_pil = Image.fromarray(cropped_rgb)

    # ── Save to cropped/ folder if filename provided ──────────────────────────
    if fname:
        save_path = os.path.join(CROP_DIR, f"{fname}_cropped.png")
        cropped_pil.save(save_path)

    new_h, new_w = cropped_bgr.shape[:2]
    return cropped_pil, cropped_bgr, (new_h, new_w)
