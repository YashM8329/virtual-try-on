"""
model_loader.py
Handles loading of all models with optimizations for 6GB VRAM (RTX 3050):
- Diffusion UNet, ControlNets, VAE → cuda:0 (Static load for maximum speed)
- Unified Preprocessor (MediaPipe) → CPU
"""

import torch
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
POSE_LANDMARKER_MODEL = "weights/pose_landmarker_heavy.task"
MP_MODEL_PATH         = "weights/selfie_multiclass_256x256.tflite"

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE_DIFF = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_unified_preprocessor():
    """Load the new Unified Preprocessor (Pose + Skin + Clothes)."""
    from unified_preprocessor import UnifiedPreprocessor
    return UnifiedPreprocessor(POSE_LANDMARKER_MODEL, MP_MODEL_PATH)

def load_openpose_generator():
    """
    Load OpenPose detector (controlnet-aux).
    Model runs on CPU — it is lightweight enough and avoids competing
    with the diffusion pipeline for VRAM.
    """
    from controlnet_aux import OpenposeDetector
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    print("✅ OpenPose generator loaded (CPU).")
    return openpose

def load_face_enhancer():
    """Load the Studio Face Enhancer (with CodeFormer AI)."""
    from face_enhancer import FaceEnhancer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return FaceEnhancer(device=device)

def load_diffusion_pipeline():
    """
    Load MultiControlNet (OpenPose + Canny) + SD Inpainting pipeline.

    OPTIMISATIONS applied
    ─────────────────────
    • Static GPU Loading (.to(DEVICE_DIFF))
        Keeping the model in VRAM provides a significant speedup for batch 
        processing compared to CPU offloading. Standard SD 1.5 + 2 ControlNets 
        fits comfortably in 6GB VRAM at FP16.

    • DPMSolverMultistepScheduler (karras sigmas)
        8-15 steps provides high quality for the neon lime hoodie.
        Cuts diffusion time by ~70% compared to standard schedulers.

    • xformers memory-efficient attention (if installed)
        ~20 % additional VRAM saving and speed-up.
    """
    pose_controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_openpose",
        torch_dtype=torch.float16,
    )
    canny_controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny",
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet=MultiControlNetModel([pose_controlnet, canny_controlnet]),
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    # ── Scheduler: DPM++ 2M Karras ────────────────────────────────────────────
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
        algorithm_type="dpmsolver++",
    )

    # ── VRAM optimizations ─────────────────────────────────────────────────────
    # Use Static GPU load for maximum speed on batch generation
    pipe.to(DEVICE_DIFF)

    # ── IP-Adapter: Reference-based conditioning ──────────────────────────────
    try:
        from transformers import CLIPVisionModelWithProjection
        
        # Load the image encoder separately to avoid subfolder path issues on Windows
        # IP-Adapter for SD 1.5 in h94/IP-Adapter expects ViT-H-14 (1024)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            torch_dtype=torch.float16,
        ).to(DEVICE_DIFF)
        
        pipe.image_encoder = image_encoder
        pipe.load_ip_adapter(
            "h94/IP-Adapter", 
            subfolder="models", 
            weight_name="ip-adapter_sd15.bin"
        )
        pipe.set_ip_adapter_scale(1.0)
        print("✅ IP-Adapter loaded (Reference-based mode active).")
    except Exception as e:
        print(f"⚠️  Failed to load IP-Adapter ({e}). Falling back to text-only.")

    # Attention slicing: reduce peak attention memory per step
    pipe.enable_attention_slicing(1)

    # xformers: ~20% speed/VRAM win when installed
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xformers memory-efficient attention enabled.")
    except Exception as e:
        print(f"⚠️  xformers not available ({e}), using default attention.")

    print(f"✅ Diffusion pipeline loaded on {DEVICE_DIFF} (Static Load).")
    return pipe
