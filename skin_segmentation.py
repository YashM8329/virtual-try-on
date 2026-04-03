"""
skin_segmentation.py
Phase 1b — MediaPipe Skin Segmentation.

Uses the selfie_multiclass_256x256 model to isolate skin/face/hair pixels.
This protection mask is later SUBTRACTED from the SAM clothing mask
to prevent neck/hand bleed-through into the generated garment.

Label map:
  0 = Background
  1 = Hair
  2 = Body / Skin
  3 = Face
  4 = Clothes
"""

import mediapipe as mp
import numpy as np


def get_skin_mask(img_rgb: np.ndarray, skin_segmenter) -> np.ndarray:
    """
    Compute a protection mask covering skin (2), face (3), and hair (1).

    Args:
        img_rgb       : np.ndarray (H, W, 3) RGB
        skin_segmenter: loaded MediaPipe ImageSegmenter instance

    Returns:
        protection_mask : np.ndarray (H, W) uint8, values 0 or 255
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = skin_segmenter.segment(mp_image)
    category_mask = result.category_mask.numpy_view()

    # Protect hair (1), body/skin (2), and face (3)
    protection_mask = np.zeros(category_mask.shape, dtype=np.uint8)
    protection_mask[
        (category_mask == 1) | (category_mask == 2) | (category_mask == 3)
    ] = 255

    return protection_mask
