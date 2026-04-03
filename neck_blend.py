"""
neck_blend.py
Neck Blending Fix — Post-processes the clothing mask to prevent
the hoodie from overlapping the neck/skin area.

Strategy:
  1. Erode the skin protection mask to get a tighter skin boundary.
  2. Subtract the eroded skin mask from the soft clothing mask.
  3. Apply a localised vertical gradient fade at the top of the
     remaining mask so the collar transitions smoothly into skin.

Usage (drop-in, no core logic change):
    from neck_blend import apply_neck_blend
    soft_mask = apply_neck_blend(soft_mask, skin_mask)
"""

import cv2
import numpy as np


def apply_neck_blend(
    soft_mask: np.ndarray,
    skin_mask: np.ndarray,
    erode_iters: int = 6,
    fade_px: int = 30,
) -> np.ndarray:
    """
    Refine soft_mask to exclude skin/neck pixels with a smooth boundary.

    Args:
        soft_mask   : (H, W) uint8 — Gaussian-blurred clothing mask from get_clothing_mask()
        skin_mask   : (H, W) uint8 — protection mask from get_skin_mask() (face+neck+hair=255)
        erode_iters : how many erosion steps to shrink the skin mask inward
                      (removes the noisy fringe at the skin/clothing boundary)
        fade_px     : vertical pixel height of the top-of-mask gradient fade

    Returns:
        refined_soft_mask : (H, W) uint8 — mask safe to pass directly to run_inpainting()
    """
    h, w = soft_mask.shape[:2]

    # ── Step 1: Erode skin mask to tighten the protected region ───────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tight_skin = cv2.erode(skin_mask, kernel, iterations=erode_iters)

    # ── Step 2: Subtract skin region from clothing mask ────────────────────────
    # Where tight_skin == 255, force mask to 0 (no generation over skin)
    result = soft_mask.copy().astype(np.float32)
    skin_f = tight_skin.astype(np.float32) / 255.0
    result = result * (1.0 - skin_f)

    # ── Step 3: Localised vertical fade at the TOP of the mask ────────────────
    # Find the topmost row where the mask is non-zero (after skin subtraction)
    col_max = result.max(axis=1)          # (H,) — max value per row
    nonzero_rows = np.where(col_max > 10)[0]

    if len(nonzero_rows) > 0:
        top_row = int(nonzero_rows[0])
        fade_end = min(top_row + fade_px, h)

        # Apply a linear alpha ramp over [top_row, fade_end]
        ramp_len = fade_end - top_row
        if ramp_len > 1:
            ramp = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)  # (ramp_len,)
            result[top_row:fade_end, :] *= ramp[:, np.newaxis]

    # ── Step 4: Clip and return ────────────────────────────────────────────────
    return np.clip(result, 0, 255).astype(np.uint8)
