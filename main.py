"""
main.py
Entry point for the Virtual Try-On pipeline.

Directory structure expected:
  input/    — source images (JPG / PNG)
  output/   — final try-on result images
  masks/    — intermediate mask debug images

Usage:
  python main.py

The script:
  1. Downloads model weights (skips if already present)
  2. Loads all models once
  3. Iterates over every image in input/
  4. For each image:
       - Extracts pose landmarks
       - Computes skin protection mask
       - Generates clothing mask (IDM-VTON style)
       - Saves mask debug composite to masks/
       - Runs ControlNet inpainting
       - Saves result to output/
"""

import os
import glob
import numpy as np
from PIL import Image

# ── Directory Setup ────────────────────────────────────────────────────────────
INPUT_DIR = "input"
OUTPUT_DIR = "output"
MASK_DIR = "masks"
SCORECARD_DIR = "scorecard"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(SCORECARD_DIR, exist_ok=True)

# ── Step 0: Download weights ───────────────────────────────────────────────────
from weight_downloader import download_weights
download_weights()

# ── Import pipeline helpers ────────────────────────────────────────────────────
from face_enhancer import FaceEnhancer
from image_utils import preprocess_image, save_images
from pose_extraction import extract_pose_landmarks, get_torso_bbox
from skin_segmentation import get_skin_mask
from clothing_mask import get_clothing_mask
from inpainting import run_inpainting
from neck_blend import apply_neck_blend
from autocrop import crop_to_hands

# ── Step 1: Load all models ────────────────────────────────────────────────────
from model_loader import load_sam, load_pose_landmarker, load_skin_segmenter, load_diffusion_pipeline

print("\n[Model Loading] Loading all models...")
sam_predictor = load_sam()
pose_detector = load_pose_landmarker()
skin_segmenter = load_skin_segmenter()
pipe = load_diffusion_pipeline()
face_enhancer = FaceEnhancer()
print("[Model Loading] All models loaded.\n")

# [NEW] Scorecard & BG removal logic
from scorecard_processor import ScorecardProcessor
import cv2


def virtual_tryon_single(image_path: str) -> Image.Image:
    """
    Full virtual try-on pipeline for a single image.

    Args:
        image_path: path to input image (JPG/PNG)

    Returns:
        PIL Image with the generated green hoodie applied
    """
    fname = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n{'='*55}")
    print(f"Processing: {image_path}")
    print(f"{'='*55}")

    # ── Step 1: Load & preprocess ──────────────────────────────────────────────
    img_pil, img_bgr, (h, w) = preprocess_image(image_path, target_size=768)
    img_rgb = np.array(img_pil)
    print(f"[1/5] Image loaded: {w}x{h}")

    # ── Step 2: Pose landmarks ─────────────────────────────────────────────────
    lm_px, annotated = extract_pose_landmarks(img_rgb, pose_detector)
    torso_bbox = get_torso_bbox(lm_px, h, w, pad_v=0.08, pad_h=0.18)
    print(f"[2/5] Pose extracted. Torso bbox: {torso_bbox}")

    # ── Auto-crop to hands (vertical crop head → lowest hand) ─────────────────
    img_pil, img_bgr, (h, w) = crop_to_hands(img_pil, img_bgr, lm_px, fname=fname)
    img_rgb = np.array(img_pil)
    print(f"      Auto-cropped → {w}x{h}")

    # ── [NEW] Face Enhancement ───────────────────────────────────────────────
    print(f"      Enhancing face (studio look)...")
    img_pil = face_enhancer.enhance_full(img_pil, bokeh_strength=15.0, save_name=fname)
    img_rgb = np.array(img_pil)
    print(f"      Enhanced image ready.")

    # ── Step 3: Skin segmentation ──────────────────────────────────────────────
    skin_mask = get_skin_mask(img_rgb, skin_segmenter)
    print("[3/5] Skin mask computed.")

    # ── Step 4: Clothing mask ──────────────────────────────────────────────────
    refined_mask, soft_mask = get_clothing_mask(img_rgb, lm_px, skin_mask, torso_bbox)
    soft_mask = apply_neck_blend(soft_mask, skin_mask)   # neck blending fix
    print("[4/5] Clothing mask refined + neck blend applied.")

    # Save mask debug composite to masks/
    import cv2
    grey_overlay = img_rgb.copy().astype(np.float32)
    alpha_arr = (refined_mask.astype(np.float32) / 255.0)[..., np.newaxis]
    grey_color = np.array([128, 128, 128], dtype=np.float32)
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

    # ── Step 5: ControlNet inpainting ─────────────────────────────────────────
    print("[5/5] Running ControlNet inpainting...")
    result_pil = run_inpainting(img_pil, soft_mask, pipe)
    print("Done!")

    return result_pil


def main():
    # Collect all JPG and PNG images from input/
    image_paths = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
        + glob.glob(os.path.join(INPUT_DIR, "*.jpeg"))
        + glob.glob(os.path.join(INPUT_DIR, "*.png"))
    )

    if not image_paths:
        print(f"⚠️  No images found in '{INPUT_DIR}/'. Please add JPG/PNG images and re-run.")
        return

    # ── [NEW] Step 2: Load Scorecard Processor ──────────────────────────────────
    print("[Scorecard] Initializing post-processing models...")
    modnet_path = "weights/modnet_photographic_portrait_matting.onnx"
    template_with_person = "template/template_1/template_1696_2528.png"
    template_bg = "template/template_1/template_not_human_1696_2528.png"
    
    scorecard_proc = ScorecardProcessor(modnet_path, template_with_person, template_bg)

    print(f"\nFound {len(image_paths)} image(s) in '{INPUT_DIR}/'.\n")

    success_count = 0
    for i, image_path in enumerate(image_paths):
        fname = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{fname}_green_hoodie.png")

        try:
            # 1. Virtual Try-On Generation
            result = virtual_tryon_single(image_path)
            result.save(out_path)
            print(f"[{i+1}/{len(image_paths)}] ✓ Try-on saved → {out_path}")

            # 2. Post-processing: BG Removal & Template Overlay
            print(f"[{i+1}/{len(image_paths)}] Creating scorecard...")
            result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            
            # Step A: Remove Background
            transparent_user = scorecard_proc.remove_background(result_bgr)
            
            # Step B: Align and Overlay onto Template
            final_scorecard = scorecard_proc.overlay_on_template(transparent_user)
            
            if final_scorecard is not None:
                sc_out_path = os.path.join(SCORECARD_DIR, f"{fname}_scorecard.png")
                cv2.imwrite(sc_out_path, final_scorecard)
                print(f"[{i+1}/{len(image_paths)}] ✓ Scorecard saved → {sc_out_path}")
            
            success_count += 1
        except Exception as e:
            print(f"[{i+1}/{len(image_paths)}] ✗ Failed — {fname}: {e}")

    print(f"\n{'='*55}")
    print(f"Completed: {success_count}/{len(image_paths)} successful")
    print(f"Results   → {OUTPUT_DIR}/")
    print(f"Masks     → {MASK_DIR}/")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
