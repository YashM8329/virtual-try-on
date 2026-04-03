# Virtual Try-On — Installation & Usage Guide

## Project Structure

```
project_root/
│
├── main.py                  # Entry point — run this
├── model_loader.py          # Loads SAM, MediaPipe, Diffusion pipeline
├── weight_downloader.py     # Downloads model weights automatically
├── image_utils.py           # Preprocessing, Canny edge, save helpers
├── pose_extraction.py       # MediaPipe pose landmark extraction
├── skin_segmentation.py     # MediaPipe skin/face/hair mask
├── clothing_mask.py         # IDM-VTON style clothing mask generation
├── inpainting.py            # ControlNet inpainting inference + prompts
├── requirements.txt         # Python dependencies
│
├── input/                   # ← Place your input images here (JPG/PNG)
├── output/                  # ← Final try-on results saved here
├── masks/                   # ← Mask debug composites saved here
└── weights/                 # ← Model weights downloaded here automatically
```

---

## Step-by-Step Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install PyTorch with CUDA support (do this FIRST)

Check your CUDA version with `nvidia-smi`, then install the matching build:

**CUDA 11.8:**
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Segment Anything Model (SAM)

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If `xformers` fails to install from requirements.txt,
> install it separately matching your CUDA version:
> ```bash
> pip install xformers --index-url https://download.pytorch.org/whl/cu118
> ```

### 5. (Important) Uninstall conflicting protobuf if present

The notebook explicitly requires:
```bash
pip uninstall -y mediapipe protobuf
pip install protobuf==4.25.3 mediapipe==0.10.14
```

---

## Usage

1. Place your input images (JPG or PNG) in the `input/` folder.
2. Run the pipeline:

```bash
python main.py
```

3. Results are saved to:
   - `output/<filename>_green_hoodie.png` — final try-on image
   - `masks/<filename>_mask_debug.png` — mask debug composite (4-panel)

---

## GPU Memory Notes

- Your shared GPU VRAM is **~7.8 GB**.
- The pipeline runs both SAM and the diffusion model on `cuda:0`
  (single GPU mode, since only one GPU is available).
- If you run out of VRAM:
  - Reduce `target_size` in `preprocess_image()` from `768` to `512`.
  - Reduce `num_inference_steps` in `DIFFUSION_CONFIG` from `50` to `30`.

---

## Weights Downloaded Automatically

| Model | Size | Source |
|---|---|---|
| SAM ViT-H | ~2.4 GB | Meta AI |
| Pose Landmarker Heavy | ~29 MB | MediaPipe |
| Selfie Multiclass | ~1 MB | MediaPipe |
| Realistic Vision V5.1 | ~2.2 GB | HuggingFace (auto) |
| ControlNet Canny | ~1.5 GB | HuggingFace (auto) |
