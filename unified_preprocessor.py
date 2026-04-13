"""
unified_preprocessor.py
Consolidates all MediaPipe and MODNet tasks into a single high-speed pass.
Targeting < 5s total preprocessing time.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class UnifiedPreprocessor:
    def __init__(self, pose_model_path, segmenter_model_path):
        # 1. Initialize Pose Landmarker (CPU is fast enough)
        base_options_pose = python.BaseOptions(model_asset_path=pose_model_path)
        options_pose = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            running_mode=vision.RunningMode.IMAGE
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)

        # 2. Initialize Image Segmenter (Selfie Multiclass)
        base_options_seg = python.BaseOptions(model_asset_path=segmenter_model_path)
        options_seg = vision.ImageSegmenterOptions(
            base_options=base_options_seg,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True
        )
        self.segmenter = vision.ImageSegmenter.create_from_options(options_seg)
        print("✅ Unified Preprocessor Initialized (Pose + Segmenter)")

    def process(self, img_rgb):
        """
        Run both models on the same image.
        Returns: landmarks (px), skin_mask, clothing_mask_raw
        """
        h, w = img_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # 1. Pose Landmarks
        pose_result = self.pose_landmarker.detect(mp_image)
        lm_px = {}
        if pose_result.pose_landmarks:
            for idx, lm in enumerate(pose_result.pose_landmarks[0]):
                lm_px[idx] = (int(lm.x * w), int(lm.y * h))
        else:
            print("⚠️ No pose landmarks detected.")

        # 2. Segmentation (Skin, Hair, Face, Clothes)
        seg_result = self.segmenter.segment(mp_image)
        category_mask = seg_result.category_mask.numpy_view()

        # Label map: 0=BG, 1=Hair, 2=Body/Skin, 3=Face, 4=Clothes
        # Protection mask (Hair + Skin + Face)
        protection_mask = np.zeros((h, w), dtype=np.uint8)
        protection_mask[(category_mask == 1) | (category_mask == 2) | (category_mask == 3)] = 255

        # Raw clothing mask (Category 4)
        clothing_mask_raw = np.zeros((h, w), dtype=np.uint8)
        clothing_mask_raw[category_mask == 4] = 255

        return lm_px, protection_mask, clothing_mask_raw

