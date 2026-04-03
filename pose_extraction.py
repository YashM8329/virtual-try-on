"""
pose_extraction.py
Phase 1a — Pose Landmark Extraction using MediaPipe Tasks API.

Key landmarks used downstream:
  Shoulders : 11, 12
  Elbows    : 13, 14
  Hips      : 23, 24
  Hands     : 15–22
"""

import mediapipe as mp
import numpy as np


def extract_pose_landmarks(img_rgb: np.ndarray, pose_detector):
    """
    Extract 33 MediaPipe Pose landmarks from an RGB image.

    Args:
        img_rgb   : np.ndarray (H, W, 3) in RGB
        pose_detector : loaded MediaPipe PoseLandmarker instance

    Returns:
        lm_px (dict {idx: (x_px, y_px)}), img_rgb (unchanged)
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = pose_detector.detect(mp_image)

    if not detection_result.pose_landmarks:
        raise ValueError("No person detected by MediaPipe.")

    h, w = img_rgb.shape[:2]
    lm_px = {}
    for idx, lm in enumerate(detection_result.pose_landmarks[0]):
        lm_px[idx] = (int(lm.x * w), int(lm.y * h))

    return lm_px, img_rgb


def get_torso_bbox(lm_px: dict, h: int, w: int, pad_v=0.15, pad_h=0.18):
    """
    Compute torso bounding box from shoulder + elbow + hip landmarks.

    Args:
        lm_px : landmark pixel dict
        h, w  : image height and width
        pad_v : vertical padding fraction
        pad_h : horizontal padding fraction

    Returns:
        (x1, y1, x2, y2)
    """
    torso_indices = [11, 12, 13, 14, 23, 24]
    xs = [lm_px[i][0] for i in torso_indices if i in lm_px]
    ys = [lm_px[i][1] for i in torso_indices if i in lm_px]
    x1 = max(0, int(min(xs) - pad_h * w))
    y1 = max(0, int(min(ys) - pad_v * h))
    x2 = min(w, int(max(xs) + pad_h * w))
    y2 = min(h, int(max(ys) + pad_v * h))
    return x1, y1, x2, y2


def get_hand_points(lm_px: dict) -> np.ndarray:
    """
    Return pixel coordinates of all hand landmarks (indices 15–22).
    Used as SAM negative prompts to exclude hands from the clothing mask.
    """
    hand_indices = [15, 16, 17, 18, 19, 20, 21, 22]
    return np.array([lm_px[i] for i in hand_indices if i in lm_px])
