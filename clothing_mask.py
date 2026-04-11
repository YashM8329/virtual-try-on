"""
clothing_mask.py
Phase 1c — Clothing Mask Generation + Mask Refinement.

IDM-VTON style mask: a wide rectangle covering the torso area.
- Top  : just below chin (derived from nose + shoulder landmarks)
- Bottom: below hips with padding
- Left/Right: nearly full image width
- Rounded top corners (cosmetic, matches IDM-VTON shape)
- Gaussian soft-edge blur for smooth boundary blending

v2 addition — Collar Extension:
- Adds a neck-hugging band above the original torso rectangle
- Covers the shoulder-to-collar transition area on both sides of the neck
- Punches out an elliptical hole around the neck itself so neck skin is preserved
- Person-adaptive: neck width derived from shoulder landmark distance
"""

import cv2
import numpy as np


def get_clothing_mask(
    img_rgb: np.ndarray,
    lm_px: dict,
    skin_mask: np.ndarray,
    torso_bbox: tuple,
    clothes_mask_raw: np.ndarray = None,
    blur_radius: int = 31,
):
    """
    Generate an IDM-VTON style clothing mask.
    """
    h, w = img_rgb.shape[:2]

    # IF RAW MASK IS PROVIDED, USE IT TO REFINE THE SEARCH AREA (Optional optimization)
    # For now, we stick to the geometric mask as it's more predictable for the AI

    # ── Top of mask: chin position ────────────────────────────────────────────
    nose_y = lm_px.get(0, (w // 2, int(h * 0.15)))[1]
    l_shoulder_y = lm_px.get(11, (w // 3, int(h * 0.40)))[1]
    r_shoulder_y = lm_px.get(12, (2 * w // 3, int(h * 0.40)))[1]
    shoulder_y = min(l_shoulder_y, r_shoulder_y)

    # Chin = 60% of the way from nose down to shoulders
    chin_y = int(nose_y + (shoulder_y - nose_y) * 0.60)
    top_y = max(int(h * 0.15), chin_y)

    # ── Bottom of mask: below hips ────────────────────────────────────────────
    l_hip_y = lm_px.get(23, (w // 3, int(h * 0.75)))[1]
    r_hip_y = lm_px.get(24, (2 * w // 3, int(h * 0.75)))[1]
    hip_y = max(l_hip_y, r_hip_y)
    bottom_y = min(h, int(hip_y + h * 0.12))

    # ── Left/Right: full image width with tiny margin ──────────────────────────
    left_x = int(w * 0.01)
    right_x = int(w * 0.99)

    # ── Build solid rectangle mask ─────────────────────────────────────────────
    refined_mask = np.zeros((h, w), dtype=np.uint8)
    refined_mask[top_y:bottom_y, left_x:right_x] = 255

    # ── Rounded top-left corner ────────────────────────────────────────────────
    corner_radius = int(w * 0.08)

    corner_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(
        corner_mask,
        (left_x, top_y),
        (left_x + corner_radius, top_y + corner_radius),
        255,
        -1,
    )
    cv2.circle(
        corner_mask,
        (left_x + corner_radius, top_y + corner_radius),
        corner_radius,
        0,
        -1,
    )
    refined_mask = cv2.subtract(refined_mask, corner_mask)

    # ── Rounded top-right corner ───────────────────────────────────────────────
    corner_mask2 = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(
        corner_mask2,
        (right_x - corner_radius, top_y),
        (right_x, top_y + corner_radius),
        255,
        -1,
    )
    cv2.circle(
        corner_mask2,
        (right_x - corner_radius, top_y + corner_radius),
        corner_radius,
        0,
        -1,
    )
    refined_mask = cv2.subtract(refined_mask, corner_mask2)

    # ── NEW: Neck/collar trapezoid mask ───────────────────────────────────────
    # Shape matches reference: wide at shoulder base, tapers to neck width
    # at chin level. Fully additive — torso mask logic above untouched.

    # Shoulder positions (base of trapezoid)
    l_shoulder_x = lm_px.get(11, (w // 3,     int(h * 0.40)))[0]
    r_shoulder_x = lm_px.get(12, (2 * w // 3, int(h * 0.40)))[0]
    l_shoulder_y = lm_px.get(11, (w // 3,     int(h * 0.40)))[1]
    r_shoulder_y = lm_px.get(12, (2 * w // 3, int(h * 0.40)))[1]
    neck_center_x = (l_shoulder_x + r_shoulder_x) // 2
    shoulder_span = abs(r_shoulder_x - l_shoulder_x)

    # Trapezoid bottom edge: at actual shoulder Y level
    trap_bottom_y = int((l_shoulder_y + r_shoulder_y) / 2)

    # Trapezoid top edge: Higher than chin_y to capture collar/neck area
    # Move from 0.60 (chin) to 0.30 (lower face/higher neck)
    trap_top_y = int(nose_y + (shoulder_y - nose_y) * 0.30)
    trap_top_y = max(0, trap_top_y)

    # Bottom width: shoulder span (full collar base)
    bottom_half_w = max(60, int(shoulder_span * 0.52))

    # Top width: neck width only (~26% of shoulder span), person-adaptive
    top_half_w = max(35, int(shoulder_span * 0.28))

    # Build trapezoid as a filled polygon
    trap_pts = np.array([
        [neck_center_x - bottom_half_w, trap_bottom_y],  # bottom-left
        [neck_center_x + bottom_half_w, trap_bottom_y],  # bottom-right
        [neck_center_x + top_half_w,    trap_top_y],     # top-right
        [neck_center_x - top_half_w,    trap_top_y],     # top-left
    ], dtype=np.int32)

    neck_trap = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(neck_trap, [trap_pts], 255)

    # Union into existing mask
    refined_mask = cv2.bitwise_or(refined_mask, neck_trap)

    # ── Skin Protection: Punch out all skin pixels ───────────────────────────
    # This ensures neck, chin, and mouth skin are preserved and not masked.
    if skin_mask is not None:
        refined_mask = cv2.bitwise_and(refined_mask, cv2.bitwise_not(skin_mask))
    
    # ── End neck/collar logic ──────────────────────────────────────────────────

    # ── Gaussian soft edge ─────────────────────────────────────────────────────
    br = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
    soft_mask = cv2.GaussianBlur(refined_mask, (br, br), sigmaX=11)

    return refined_mask, soft_mask