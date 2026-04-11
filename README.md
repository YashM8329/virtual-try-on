# Virtual Try-On & Scorecard Generator

A high-fidelity Virtual Try-On pipeline designed to replace clothing (specifically optimized for a neon lime green hoodie) using Generative AI and advanced computer vision. The system is highly optimized to process images in **under 30 seconds** on consumer-grade hardware (e.g., RTX 3050).

## 🚀 Key Features

- **High-Speed Inference**: Optimized to process one image in **~25 seconds** on a 6GB VRAM GPU.
- **Unified Preprocessing**: Consolidates pose extraction, skin segmentation, and clothing identification into a single high-speed pass.
- **Precision Masking**: Utilizes IDM-VTON style masking with neck and collar protection to ensure natural garment transitions.
- **Fast Diffusion Engine**: Leverages 8-step DPM++ inpainting at 512px for maximum speed without quality loss.
- **Studio-Grade Face Enhancement**: Landmark-based face polish that skips redundant detection steps.
- **One-Pass Background Removal**: Caches initial alpha masks to eliminate redundant MODNet calls in the post-processing phase.
- **VRAM Optimized**: Tuned for 6GB+ VRAM using static memory management and `xformers`.

## 🛠️ Architecture & Workflow

### 1. Initialization
The system requires pre-trained weights for MediaPipe, GFPGAN, and MODNet.
```bash
python weight_downloader.py
```

### 2. Processing Pipeline (`main.py`)
- **Preprocessing**: One MediaPipe pass for Pose + Skin + Clothes.
- **Masking**: Adaptive IDM-VTON style masking refined by skin protection.
- **BG Removal**: MODNet isolations cached for final scorecard composition.
- **Diffusion**: 8-step DPM++ inpainting using Multi-ControlNet (OpenPose + Canny).
- **Refinement**: Landmark-based face polish and studio lighting effects.
- **Composition**: Branded scorecard generation using cached transparency masks.

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ftb-scorecard
   ```

2. **Setup Environment**:
   It is recommended to use a virtual environment with Python 3.12.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download Weights**:
   ```bash
   python weight_downloader.py
   ```

## 🏃 Usage

1. Place your target images in the `input/` folder (supported: `.png`, `.jpg`, `.jpeg`).
2. Run the main processing script:
   ```bash
   python main.py
   ```
3. Results are saved in real-time as they are processed.

### Output Locations:
- `output/`: Final high-resolution try-on results.
- `masks/`: Debugging views of generated clothing and skin masks.
- `enhanced/`: Face-enhanced versions of the outputs.
- `scorecard/`: Final branded scorecards.

## 🗄️ Project Structure

- `main.py`: Entry point orchestrating the high-speed pipeline.
- `unified_preprocessor.py`: Consolidated MediaPipe logic.
- `inpainting.py`: Core diffusion logic with speed optimizations.
- `clothing_mask.py`: Advanced masking logic for cloth replacement.
- `face_enhancer.py`: Studio-quality face and skin refinement.
- `scorecard_processor.py`: Background removal and template composition.
- `model_loader.py`: Hardware-specific optimizations for VRAM management.

## ⚙️ Technical Requirements

- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 3050 or better recommended).
- **RAM**: 16GB Minimum.
- **Storage**: ~5GB for models and dependencies.
- **OS**: Windows (tested) / Linux.

## 📄 License
This project is for research and development purposes in generative AI.
