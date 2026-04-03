"""
main.py
Entry point for the Virtual Try-On pipeline.

Directory structure:
  input/    — source images (JPG / PNG)
  output/   — final try-on result images
  masks/    — intermediate mask debug images
  weights/  — model weights (auto-downloaded)

Usage:
  python main.py
"""

import os
import gc
import glob
import torch
import numpy as np
from PIL import Image

# ── Directory Setup ────────────────────────────────────────────────────────────
INPUT_DIR  = "input"
OUTPUT_DIR = "output"
MASK_DIR   = "masks"

os.makedirs(INPUT_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_DIR,   exist_ok=True)

# ── Step 0: Download weights ───────────────────────────────────────────────────
from weight_downloader import download_weights
download_weights()

# ── Step 1: Load CPU-based models (always in memory) ──────────────────────────
from model_loader import (
    load_sam, load_pose_landmarker, load_skin_segmenter,
    load_diffusion_pipeline, DEVICE_SAM
)

print("\n[Model Loading] Loading MediaPipe models (CPU)...")
pose_detector  = load_pose_landmarker()
skin_segmenter = load_skin_segmenter()

# ── Step 2: Load SAM (GPU — will be offloaded before diffusion) ────────────────
print("[Model Loading] Loading SAM on GPU...")
sam_model, sam_predictor = load_sam()

# ── Step 3: Load Diffusion pipeline (CPU offload mode) ────────────────────────
print("[Model Loading] Loading Diffusion pipeline (CPU offload mode)...")
pipe = load_diffusion_pipeline()
print("[Model Loading] All models ready.\n")

# ── Import pipeline helpers ────────────────────────────────────────────────────
from image_utils      import preprocess_image, save_images
from pose_extraction  import extract_pose_landmarks, get_torso_bbox
from skin_segmentation import get_skin_mask
from clothing_mask    import get_clothing_mask
from inpainting       import run_inpainting


def free_sam_vram():
    """Move SAM to CPU and free VRAM before running diffusion."""
    sam_model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    print("      SAM offloaded to CPU — VRAM freed for diffusion.")


def restore_sam_vram():
    """Move SAM back to GPU for next image's masking step."""
    sam_model.to(DEVICE_SAM)
    torch.cuda.empty_cache()
    print("      SAM restored to GPU.")


def virtual_tryon_single(image_path: str) -> Image.Image:
    """Full virtual try-on pipeline for a single image."""
    fname = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n{'='*55}")
    print(f"Processing: {image_path}")
    print(f"{'='*55}")

    # ── Step 1: Load & preprocess ──────────────────────────────────────────────
    img_pil, img_bgr, (h, w) = preprocess_image(image_path, target_size=512)
    img_rgb = np.array(img_pil)
    print(f"[1/5] Image loaded: {w}x{h}")

    # ── Step 2: Pose landmarks ─────────────────────────────────────────────────
    lm_px, annotated = extract_pose_landmarks(img_rgb, pose_detector)
    torso_bbox = get_torso_bbox(lm_px, h, w, pad_v=0.08, pad_h=0.18)
    print(f"[2/5] Pose extracted. Torso bbox: {torso_bbox}")

    # ── Step 3: Skin segmentation ──────────────────────────────────────────────
    skin_mask = get_skin_mask(img_rgb, skin_segmenter)
    print("[3/5] Skin mask computed.")

    # ── Step 4: Clothing mask ──────────────────────────────────────────────────
    refined_mask, soft_mask = get_clothing_mask(img_rgb, lm_px, skin_mask, torso_bbox)
    print("[4/5] Clothing mask refined.")

    # Save mask debug composite to masks/
    import cv2
    grey_overlay = img_rgb.copy().astype(np.float32)
    alpha_arr    = (refined_mask.astype(np.float32) / 255.0)[..., np.newaxis]
    grey_color   = np.array([128, 128, 128], dtype=np.float32)
    grey_overlay = (grey_color * alpha_arr + grey_overlay * (1 - alpha_arr)).astype(np.uint8)

    mask_debug_path = os.path.join(MASK_DIR, f"{fname}_mask_debug.png")
    save_images(
        {
            "Original": img_rgb,
            "Refined Binary Mask": refined_mask,
            "Soft Mask (Gaussian)": soft_mask,
            "Mask Overlay (IDM style)": grey_overlay,
        },
        save_path=mask_debug_path,
        figsize=(18, 6),
    )
    print(f"      Mask debug saved → {mask_debug_path}")

    # ── VRAM management: offload SAM before diffusion ──────────────────────────
    free_sam_vram()

    # ── Step 5: ControlNet inpainting ─────────────────────────────────────────
    print("[5/5] Running ControlNet inpainting...")
    result_pil = run_inpainting(img_pil, soft_mask, pipe)
    print("Done!")

    # ── Restore SAM for next image ─────────────────────────────────────────────
    restore_sam_vram()

    return result_pil


def main():
    image_paths = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
        + glob.glob(os.path.join(INPUT_DIR, "*.jpeg"))
        + glob.glob(os.path.join(INPUT_DIR, "*.png"))
    )

    if not image_paths:
        print(f"⚠️  No images found in '{INPUT_DIR}/'. Add JPG/PNG images and re-run.")
        return

    print(f"\nFound {len(image_paths)} image(s) in '{INPUT_DIR}/'.\n")

    success_count = 0
    for i, image_path in enumerate(image_paths):
        fname    = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{fname}_green_hoodie.png")

        try:
            result = virtual_tryon_single(image_path)
            result.save(out_path)
            print(f"[{i+1}/{len(image_paths)}] ✓ Saved → {out_path}")
            success_count += 1
        except Exception as e:
            print(f"[{i+1}/{len(image_paths)}] ✗ Failed — {fname}: {e}")
            # Free any leftover VRAM on failure
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\n{'='*55}")
    print(f"Completed : {success_count}/{len(image_paths)} successful")
    print(f"Results   → {OUTPUT_DIR}/")
    print(f"Masks     → {MASK_DIR}/")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()