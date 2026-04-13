"""
clothing_mask.py
Pipeline step 4 — Clothing Mask Generation.

Builds the inpainting mask that defines where the hoodie will be generated.
Two shapes are unioned and then skin pixels are subtracted:

  Torso rectangle
    Top   : chin level  (60 % of the way from nose landmark 0 to shoulders 11/12)
    Bottom: below hips  (landmarks 23/24 + 12 % height padding)
    Sides : near full image width (1 % margin each side)
    Corners: rounded to soften the top edge

  Collar trapezoid
    Wide at shoulder baseline, tapers to neck width at collar height.
    Covers the hoodie collar / shoulder-to-neck transition area.
    Width is person-adaptive (derived from shoulder span).

After the union, the skin protection mask (face + hair + skin from MediaPipe)
is subtracted so neck, chin, and face pixels are never inpainted.

Output: hard binary `refined_mask` + Gaussian-blurred `soft_mask` (used for
        smooth alpha compositing in run_inpainting).
"""

import cv2
import numpy as np


def get_clothing_mask(
    img_rgb: np.ndarray,
    lm_px: dict,
    skin_mask: np.ndarray,
    blur_radius: int = 31,
):
    """
    Build the hoodie inpainting mask from MediaPipe pose landmarks.

    Args:
        img_rgb    : (H, W, 3) uint8 — preprocessed input image (1024px).
        lm_px      : landmark pixel dict from UnifiedPreprocessor.process().
                     Keys used: 0 (nose), 11/12 (shoulders), 23/24 (hips).
        skin_mask  : (H, W) uint8 — protection mask (face + hair + skin = 255)
                     from UnifiedPreprocessor; subtracted to keep skin unmasked.
        blur_radius: kernel size for Gaussian soft-edge blur on the final mask.

    Returns:
        refined_mask : (H, W) uint8 — hard binary mask.
        soft_mask    : (H, W) uint8 — Gaussian-blurred mask for alpha compositing.
    """
    h, w = img_rgb.shape[:2]

    # ── Torso rectangle bounds ─────────────────────────────────────────────────
    # Top: 60 % of the way from nose (lm 0) down to the higher shoulder (lm 11/12)
    nose_y = lm_px.get(0, (w // 2, int(h * 0.15)))[1]
    l_shoulder_y = lm_px.get(11, (w // 3, int(h * 0.40)))[1]
    r_shoulder_y = lm_px.get(12, (2 * w // 3, int(h * 0.40)))[1]
    shoulder_y = min(l_shoulder_y, r_shoulder_y)
    chin_y = int(nose_y + (shoulder_y - nose_y) * 0.60)
    top_y = max(int(h * 0.15), chin_y)

    # Bottom: below hips (lm 23/24) with 12 % height padding
    l_hip_y = lm_px.get(23, (w // 3, int(h * 0.75)))[1]
    r_hip_y = lm_px.get(24, (2 * w // 3, int(h * 0.75)))[1]
    hip_y = max(l_hip_y, r_hip_y)
    bottom_y = min(h, int(hip_y + h * 0.12))

    # Sides: near full image width (1 % margin keeps away from frame edge)
    left_x = int(w * 0.01)
    right_x = int(w * 0.99)

    # ── Solid torso rectangle ──────────────────────────────────────────────────
    refined_mask = np.zeros((h, w), dtype=np.uint8)
    refined_mask[top_y:bottom_y, left_x:right_x] = 255

    # ── Round the top corners (softens the hard rectangle edge) ───────────────
    corner_radius = int(w * 0.08)

    corner_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(corner_mask, (left_x, top_y),
                  (left_x + corner_radius, top_y + corner_radius), 255, -1)
    cv2.circle(corner_mask, (left_x + corner_radius, top_y + corner_radius),
               corner_radius, 0, -1)
    refined_mask = cv2.subtract(refined_mask, corner_mask)

    corner_mask2 = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(corner_mask2, (right_x - corner_radius, top_y),
                  (right_x, top_y + corner_radius), 255, -1)
    cv2.circle(corner_mask2, (right_x - corner_radius, top_y + corner_radius),
               corner_radius, 0, -1)
    refined_mask = cv2.subtract(refined_mask, corner_mask2)

    # ── Collar trapezoid — covers hoodie collar / shoulder-to-neck transition ──
    # Base spans the full shoulder width; top narrows to neck width.
    # Person-adaptive: both widths scale with the shoulder landmark distance.
    l_shoulder_x = lm_px.get(11, (w // 3,     int(h * 0.40)))[0]
    r_shoulder_x = lm_px.get(12, (2 * w // 3, int(h * 0.40)))[0]
    l_shoulder_y = lm_px.get(11, (w // 3,     int(h * 0.40)))[1]
    r_shoulder_y = lm_px.get(12, (2 * w // 3, int(h * 0.40)))[1]
    neck_center_x = (l_shoulder_x + r_shoulder_x) // 2
    shoulder_span = abs(r_shoulder_x - l_shoulder_x)

    trap_bottom_y = int((l_shoulder_y + r_shoulder_y) / 2)   # shoulder baseline
    trap_top_y    = max(0, int(nose_y + (shoulder_y - nose_y) * 0.30))  # collar top

    bottom_half_w = max(60, int(shoulder_span * 0.52))   # full collar base
    top_half_w    = max(35, int(shoulder_span * 0.28))   # neck width (~28 % of span)

    trap_pts = np.array([
        [neck_center_x - bottom_half_w, trap_bottom_y],  # bottom-left
        [neck_center_x + bottom_half_w, trap_bottom_y],  # bottom-right
        [neck_center_x + top_half_w,    trap_top_y],     # top-right
        [neck_center_x - top_half_w,    trap_top_y],     # top-left
    ], dtype=np.int32)

    neck_trap = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(neck_trap, [trap_pts], 255)
    refined_mask = cv2.bitwise_or(refined_mask, neck_trap)

    # ── Subtract skin pixels so neck/chin/face are never inpainted ────────────
    if skin_mask is not None:
        refined_mask = cv2.bitwise_and(refined_mask, cv2.bitwise_not(skin_mask))

    # ── Gaussian soft edge for smooth alpha compositing in run_inpainting ─────
    br = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
    soft_mask = cv2.GaussianBlur(refined_mask, (br, br), sigmaX=11)

    return refined_mask, soft_mask