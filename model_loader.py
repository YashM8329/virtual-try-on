"""
model_loader.py
Handles loading of all models:
- SAM ViT-H (cuda:0 or cpu)
- MediaPipe Pose Landmarker (CPU)
- MediaPipe Skin Segmenter (CPU)
- ControlNet + Stable Diffusion Inpainting Pipeline (cuda:0 or cpu)
"""

import os
import torch
import mediapipe as mp
from segment_anything import sam_model_registry, SamPredictor
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
SAM_CHECKPOINT = "weights/sam_vit_h_4b8939.pth"
POSE_LANDMARKER_MODEL = "weights/pose_landmarker_heavy.task"
MP_MODEL_PATH = "weights/selfie_multiclass_256x256.tflite"

# ── Device Selection ───────────────────────────────────────────────────────────
# With a single shared GPU of ~7.8 GB, both SAM and diffusion run on cuda:0
DEVICE_SAM = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_DIFF = "cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")


def load_sam():
    """Load Segment Anything Model (SAM ViT-H)."""
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE_SAM)
    sam_predictor = SamPredictor(sam)
    print(f"✅ SAM loaded on {DEVICE_SAM}")
    return sam_predictor


def load_pose_landmarker():
    """Load MediaPipe Pose Landmarker (Tasks API)."""
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_LANDMARKER_MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    pose_detector = PoseLandmarker.create_from_options(pose_options)
    print("✅ MediaPipe Pose Landmarker loaded.")
    return pose_detector


def load_skin_segmenter():
    """Load MediaPipe Selfie Multiclass Segmenter."""
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions

    segmenter_options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
        output_category_mask=True,
    )
    skin_segmenter = ImageSegmenter.create_from_options(segmenter_options)
    print("✅ MediaPipe Skin Segmenter loaded.")
    return skin_segmenter


def load_diffusion_pipeline():
    """Load ControlNet + Stable Diffusion ControlNet Inpainting Pipeline."""
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=torch.float16,
    ).to(DEVICE_DIFF)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(DEVICE_DIFF)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        print("[Model Loading] xformers not found or incompatible. Clipping to default attention optimizations.")
        pass
    print(f"✅ Diffusion pipeline loaded on {DEVICE_DIFF}")
    return pipe
