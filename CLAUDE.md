# CLAUDE.md

This file provides comprehensive guidance for developers and AI agents working on the **ftb-scorecard** project.

## Project Overview (Agent Perspective)
- **Goal**: High-speed (< 30s/image) virtual try-on system that replaces a person's top with a specific neon lime green hoodie and generates a branded scorecard.
- **Core Architecture**: Modular pipeline combining MediaPipe (Pose/Segmentation), MODNet (Matting), Stable Diffusion 1.5 (Inpainting via ControlNet + IP-Adapter), and CodeFormer (Face Restoration) + Classical CV (Studio Polish).
- **Performance Target**: Optimized for 6GB VRAM (RTX 3050) using FP16, static GPU loading, and DPM++ scheduling.
- **Key Strategy**: All heavy models are loaded once at startup. Intermediate results like landmarks and background masks are cached/reused to minimize redundant computation.

---

## Libraries & Technologies

### Core Deep Learning & Generative AI
- **PyTorch & Torchvision**: Foundation for model execution and image transformations.
- **Diffusers**: High-level API for Stable Diffusion 1.5 and ControlNet pipelines.
- **Transformers & Accelerate**: HuggingFace utilities for model loading and optimization.
- **xformers**: Memory-efficient attention for reduced VRAM usage.
- **ControlNet-Aux**: Pre-processing tools for Canny and OpenPose.
- **SafeTensors**: Secure and fast weight loading.

### Computer Vision & AI Frameworks
- **MediaPipe**: Real-time Pose Landmarking and Selfie Segmentation.
- **OpenCV (cv2)**: Essential image processing, drawing, and classical CV filters.
- **ONNX Runtime**: Optimized inference for MODNet background removal.
- **Pillow (PIL)**: Standard image handling and editing.
- **Scikit-image & Scipy**: Advanced image processing and mathematical operations.

### Utilities
- **NumPy**: Numerical computing and array manipulations.
- **HuggingFace Hub**: Automated model weight management.
- **Omegaconf & Einops**: Configuration management and flexible tensor operations.
- **Tqdm**: Progress tracking for long-running processes.

---

## Detailed Pipeline Flow & Parameters

### 1. Preprocessing & Unified Pass
- **Module**: `unified_preprocessor.py`
- **Action**: Resizes input to `diffusion_size` (1024px) and runs MediaPipe Pose Landmarker + Selfie Multiclass Segmenter in a single pass.
- **Key Parameters**:
  - `model_complexity=1` (MediaPipe Pose)
  - `target_size=1024`
  - **Output**: `lm_px` (pose landmarks), `skin_mask`, `clothes_mask_raw`.

### 2. Background Extraction (Parallel/Cached)
- **Module**: `scorecard_processor.py` (via MODNet)
- **Action**: Extracts high-quality alpha mask for the person. Cached for the final scorecard step.
- **Key Parameters**: `resolution=(512, 512)` (Internal MODNet inference).

### 3. Clothing Mask Generation
- **Module**: `clothing_mask.py`
- **Action**: Builds an IDM-VTON style torso rectangle + trapezoid collar extension using landmarks, then subtracts skin.
- **Key Parameters**:
  - Landmark Indices: 11/12 (Shoulders), 23/24 (Hips), 0 (Nose).
  - `trapezoid_height_factor=0.4` (Collar height relative to neck).
  - **Output**: `refined_mask` (hard), `soft_mask` (Gaussian blurred).

### 4. Neck Blending
- **Module**: `neck_blend.py`
- **Action**: Erodes skin mask and subtracts it from the soft mask with a linear fade to ensure seamless garment-to-skin transitions.
- **Key Parameters**: `erode_kernel=5`, `fade_height=50`.

### 5. Multi-ControlNet Inpainting
- **Module**: `inpainting.py`
- **Action**: Stable Diffusion 1.5 with OpenPose and Canny ControlNets. Uses IP-Adapter for garment reference.
- **Key Parameters**:
  - `diffusion_size=768` (Inference resolution for speed/VRAM).
  - `num_inference_steps=15-20` (DPM++ 2M Karras).
  - `guidance_scale=8.0 - 10.0`.
  - `controlnet_conditioning_scale=[0.7, 0.4]` (OpenPose, Canny).
  - `ip_adapter_scale=1.0`.

### 6. Face Enhancement (Studio Polish)
- **Module**: `face_enhancer.py`
- **Action**: Classical CV steps (Bilateral filter, Frequency separation, CLAHE, Glow, Sharpen) using cached landmarks.
- **Key Parameters**:
  - `bilateral_sigma=20` (Skin smoothing).
  - `unsharp_strength=1.5` (Feature sharpening).
  - `glow_opacity=0.3`.

### 7. Scorecard Compositing
- **Module**: `scorecard_processor.py`
- **Action**: Aligns the enhanced result onto a template using shoulder landmark indices (11, 12).
- **Template Path**: `template/template_1/`.

---

## Installation & Execution

### Installation Order (Critical)
1. **PyTorch with CUDA**: `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118`
2. **SAM**: `pip install git+https://github.com/facebookresearch/segment-anything.git`
3. **Dependencies**: `pip install -r requirements.txt`
4. **Protobuf Fix**: `pip uninstall -y mediapipe protobuf && pip install protobuf==4.25.3 mediapipe==0.10.14`

### Running the Pipeline
```bash
# Full pipeline
python main.py

# Standalone Face Enhancement
python face_enhancer.py input.jpg output.jpg

# Weight Downloader
python weight_downloader.py
```

---

## VRAM & Optimization Tips
- **RTX 3050 (6GB)**: Use `diffusion_size=768` and `num_inference_steps=15`.
- **Higher VRAM**: Increase `diffusion_size` to `1024` and `steps` to `30`.
- **IP-Adapter**: Ensure `garment.jpg` is in the root directory. If missing, the system defaults to text-prompt only.
- **CPU Offloading**: Disabled by default for batch speed. Enable in `model_loader.py` if VRAM is < 4GB.

---

## Project Directory Structure
```text
C:\Users\yashm\Desktop\ftb-scorecard\
├── main.py                 # Main entry point & orchestration
├── model_loader.py         # VRAM-optimized model initialization
├── inpainting.py           # SD 1.5 + MultiControlNet + IP-Adapter logic
├── unified_preprocessor.py # MediaPipe Pose + Segmentation integration
├── clothing_mask.py        # Geometric mask generation logic
├── neck_blend.py           # Alpha blending for neck/garment transitions
├── face_enhancer.py        # Classical CV portrait enhancement
├── scorecard_processor.py   # MODNet BG removal & Template compositing
├── image_utils.py          # Image IO, resizing, and Canny/OpenPose pre-processing
├── weight_downloader.py    # Automated weight fetching (HuggingFace/Local)
├── requirements.txt        # Project dependencies
├── garment.png             # Reference garment image (Hoodie)
├── input/                  # Input images (JPG/PNG)
├── output/                 # Virtual try-on results
├── scorecard/              # Final composite scorecards
├── masks/                  # Debug masks (Garment, Skin, BG)
├── template/               # Branded background templates
├── weights/                # Model weights (.safetensors, .onnx, .bin)
└── __pycache__/            # Python bytecode
```
tensors, .onnx, .bin)
└── __pycache__/            # Python bytecode
```
