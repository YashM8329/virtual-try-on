"""
model_loader.py
Handles loading of all models with 6 GB VRAM optimizations:
- SAM ViT-H       → cuda:0  (loaded, then offloaded after mask generation)
- MediaPipe Pose  → CPU
- MediaPipe Skin  → CPU
- ControlNet + SD Inpainting → cuda:0 with CPU offload + attention slicing
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
SAM_CHECKPOINT      = "weights/sam_vit_h_4b8939.pth"
POSE_LANDMARKER_MODEL = "weights/pose_landmarker_heavy.task"
MP_MODEL_PATH       = "weights/selfie_multiclass_256x256.tflite"

# ── Device — single GPU (RTX 3050, 6 GB) ──────────────────────────────────────
DEVICE_SAM  = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_DIFF = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_sam():
    """Load SAM ViT-H. Will be moved to CPU after masking to free VRAM."""
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(DEVICE_SAM)
    sam_predictor = SamPredictor(sam)
    print(f"✅ SAM loaded on {DEVICE_SAM}")
    return sam, sam_predictor   # return sam model too so we can offload it


def load_pose_landmarker():
    """Load MediaPipe Pose Landmarker (Tasks API, CPU)."""
    BaseOptions         = mp.tasks.BaseOptions
    PoseLandmarker      = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_LANDMARKER_MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    pose_detector = PoseLandmarker.create_from_options(pose_options)
    print("✅ MediaPipe Pose Landmarker loaded.")
    return pose_detector


def load_skin_segmenter():
    """Load MediaPipe Selfie Multiclass Segmenter (CPU)."""
    BaseOptions          = mp.tasks.BaseOptions
    ImageSegmenter       = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions

    segmenter_options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
        output_category_mask=True,
    )
    skin_segmenter = ImageSegmenter.create_from_options(segmenter_options)
    print("✅ MediaPipe Skin Segmenter loaded.")
    return skin_segmenter


def load_diffusion_pipeline():
    """
    Load ControlNet + SD Inpainting pipeline with 6 GB VRAM optimizations:
      - enable_model_cpu_offload : moves layers to CPU when not in use
      - enable_attention_slicing : cuts peak attention memory
      - xformers memory efficient attention (if available)
    NOTE: do NOT call .to(device) when using enable_model_cpu_offload —
    the pipeline manages device placement automatically.
    """
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # ── 6 GB VRAM optimizations ────────────────────────────────────────────────
    # CPU offload: streams model layers to GPU only when needed
    pipe.enable_model_cpu_offload()
    # Attention slicing: reduces peak VRAM during cross-attention
    pipe.enable_attention_slicing(1)
    # xformers: memory-efficient attention kernels (use if installed correctly)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xformers memory-efficient attention enabled.")
    except Exception as e:
        print(f"⚠️  xformers not available ({e}), using default attention.")

    print(f"✅ Diffusion pipeline loaded with CPU offload (VRAM-safe mode).")
    return pipe