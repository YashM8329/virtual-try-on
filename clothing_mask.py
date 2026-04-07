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
    blur_radius: int = 31,
):
    """
    Generate an IDM-VTON style clothing mask.

    Args:
        img_rgb    : np.ndarray (H, W, 3) RGB
        lm_px      : landmark pixel dict from pose extraction
        skin_mask  : protection mask (H, W) uint8 from skin segmentation
        torso_bbox : (x1, y1, x2, y2) torso bounding box (unused directly,
                     kept for API parity with notebook)
        blur_radius: Gaussian blur kernel size for soft edge

    Returns:
        refined_mask : np.ndarray (H, W) uint8 — hard binary mask
        soft_mask    : np.ndarray (H, W) uint8 — Gaussian-blurred soft mask
    """
    h, w = img_rgb.shape[:2]

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

    # ── NEW: Collar extension — neck-hugging band above the torso rectangle ───
    # Adds coverage for the shirt collar area on both sides of the neck while
    # preserving the neck skin. Fully additive — original logic above untouched.

    # Shoulder landmarks: X for neck center, Y for shoulder line
    l_shoulder_x = lm_px.get(11, (w // 3,     int(h * 0.40)))[0]
    r_shoulder_x = lm_px.get(12, (2 * w // 3, int(h * 0.40)))[0]
    neck_center_x = (l_shoulder_x + r_shoulder_x) // 2
    shoulder_span = abs(r_shoulder_x - l_shoulder_x)

    # Neck half-width: ~24% of shoulder span, person-adaptive, min 40px
    neck_half_w = max(40, int(shoulder_span * 0.24))

    # Collar band height: fixed at 8% of image height so it's always visible
    # regardless of how close the chin is to the shoulders in the photo
    collar_band_height = max(int(h * 0.08), 40)
    collar_top_y = max(0, top_y - collar_band_height)

    # Step 1 — full-width rectangle covering the collar zone
    collar_ext = np.zeros((h, w), dtype=np.uint8)
    collar_ext[collar_top_y:top_y, left_x:right_x] = 255

    # Step 2 — ellipse cutout to preserve the neck
    #   The ellipse is centered at top_y (the seam between collar band and
    #   existing torso mask) so the bottom half of the ellipse cuts into the
    #   collar band while the top half cuts into nothing — giving a clean
    #   U-shaped notch whose depth == ellipse vertical radius.
    #   Vertical radius = collar_band_height * 0.85 so the notch nearly fills
    #   the band height, leaving only the sides masked.
    ellipse_center = (neck_center_x, top_y)
    ellipse_axes   = (neck_half_w, int(collar_band_height * 0.85))
    cv2.ellipse(
        collar_ext,
        ellipse_center,
        ellipse_axes,
        angle=0,
        startAngle=180,    # upper half of ellipse → U-notch opening upward
        endAngle=360,
        color=0,
        thickness=-1,
    )

    # Step 3 — union into the existing mask (purely additive)
    refined_mask = cv2.bitwise_or(refined_mask, collar_ext)
    # ── End collar extension ───────────────────────────────────────────────────

    # ── Gaussian soft edge ─────────────────────────────────────────────────────
    br = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
    soft_mask = cv2.GaussianBlur(refined_mask, (br, br), sigmaX=11)

    return refined_mask, soft_mask