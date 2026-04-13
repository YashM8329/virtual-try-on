"""
main.py
Entry point for the Optimized Virtual Try-On pipeline.
Target: < 30 seconds per image.
"""

import os
import glob
import cv2
import numpy as np
import warnings
from PIL import Image

# Suppress noisy library warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging

# ── Directory Setup ────────────────────────────────────────────────────────────
INPUT_DIR    = "input"
OUTPUT_DIR   = "output"
MASK_DIR     = "masks"
SCORECARD_DIR = "scorecard"

for d in (INPUT_DIR, OUTPUT_DIR, MASK_DIR, SCORECARD_DIR):
    os.makedirs(d, exist_ok=True)

# ── Step 0: Download weights ───────────────────────────────────────────────────
from weight_downloader import download_weights
download_weights()

# ── Import pipeline helpers ────────────────────────────────────────────────────
from face_enhancer    import FaceEnhancer
from image_utils      import preprocess_image
from unified_preprocessor import UnifiedPreprocessor
from clothing_mask    import get_clothing_mask
from inpainting       import run_inpainting
from neck_blend       import apply_neck_blend
from scorecard_processor import ScorecardProcessor

# ── Step 1: Load all models once ──────────────────────────────────────────────
from model_loader import (
    load_unified_preprocessor,
    load_diffusion_pipeline, load_openpose_generator,
    load_face_enhancer,
)

print("\n[Model Loading] Loading all models...")
preprocessor   = load_unified_preprocessor()
openpose_gen   = load_openpose_generator()
pipe           = load_diffusion_pipeline()
face_enhancer  = load_face_enhancer()
print("[Model Loading] All models loaded.\n")

# ── Scorecard processor ────────────────────────────────────────────────────────
print("[Scorecard] Initializing post-processing models...")
scorecard_proc = ScorecardProcessor(
    "weights/modnet_photographic_portrait_matting.onnx",
    "template/template_1/template_1696_2528.png",
    "template/template_1/template_not_human_1696_2528.png",
)

def main():
    image_paths = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
        + glob.glob(os.path.join(INPUT_DIR, "*.jpeg"))
        + glob.glob(os.path.join(INPUT_DIR, "*.png"))
    )

    if not image_paths:
        print(f"⚠️  No images found in '{INPUT_DIR}/'. Add JPG/PNG images and re-run.")
        return

    print(f"\nFound {len(image_paths)} image(s) in '{INPUT_DIR}/'.")

    success_count = 0
    # Higher quality: 20 steps, 768px resolution (fixes blurriness)
    config = {"num_inference_steps": 20, "diffusion_size": 1024, "guidance_scale": 10.0} 

    # Load Reference Garment (IP-Adapter)
    garment_path = "garment.png"

    garment_pil = None
    if os.path.exists(garment_path):
        garment_pil = Image.open(garment_path).convert("RGB")
        print(f"✅ Reference garment loaded: {garment_path}")
    else:
        print(f"⚠️  Reference garment not found at '{garment_path}'. Falling back to prompt-only.")

    for i, image_path in enumerate(image_paths):
        fname = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{fname}_green_hoodie.png")
        print(f"\n[{i+1}/{len(image_paths)}] Processing {fname}...")

        try:
            # 1. Precompute (Unified Pass)
            img_pil, img_bgr, (h, w) = preprocess_image(image_path, target_size=1024)
            img_rgb = np.array(img_pil)
            
            # Run Pose + Skin + Clothes Segmenter in ONE pass (Saves ~10s)
            lm_px, skin_mask, clothes_mask_raw = preprocessor.process(img_rgb)

            # BG Removal - Cache the BGRA result (Saves ~15s later)
            bgra_original = scorecard_proc.remove_background(img_bgr)
            alpha_orig = bgra_original[:, :, 3] / 255.0
            img_bgr_white = (bgra_original[:, :, :3] * alpha_orig[:, :, np.newaxis] + (1 - alpha_orig[:, :, np.newaxis]) * 255).astype(np.uint8)
            img_pil_white = Image.fromarray(cv2.cvtColor(img_bgr_white, cv2.COLOR_BGR2RGB))

            # Masking logic
            refined_mask, soft_mask = get_clothing_mask(img_rgb, lm_px, skin_mask)
            soft_mask = apply_neck_blend(soft_mask, skin_mask)

            # 2. Diffusion (Inpainting with optional Reference Image)
            result = run_inpainting(
                img_pil=img_pil_white,
                soft_mask=soft_mask,
                pipe=pipe,
                config=config,
                openpose_gen=openpose_gen,
                image_path=image_path,
                garment_image_pil=garment_pil
            )
            result.save(out_path)
            print(f"  ✓ Try-on saved")

            # 3. Final Polish & Scorecard
            # Use cached alpha mask instead of re-running MODNet (Saves 10-15s)
            result_np = np.array(result)
            
            # Landmark-based Face Enhancement (landmark reuse saves 2s)
            enhanced_pil = face_enhancer.enhance_full(
                Image.fromarray(result_np),
                save_name=fname,
                user_landmarks=lm_px,
                skin_mask=skin_mask,
            )
            enhanced_bgr = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
            
            # Final composite using cached alpha
            final_composite = np.concatenate([enhanced_bgr, bgra_original[:, :, 3:4]], axis=2)
            final_scorecard = scorecard_proc.overlay_on_template(final_composite, user_landmarks=lm_px)

            if final_scorecard is not None:
                sc_path = os.path.join(SCORECARD_DIR, f"{fname}_scorecard.png")
                cv2.imwrite(sc_path, final_scorecard)
                print(f"  ✓ Scorecard saved")

            success_count += 1

        except Exception as e:
            print(f"  ✗ Failed — {fname}: {e}")

    print(f"\n{'='*55}")
    print(f"Completed : {success_count}/{len(image_paths)} successful")
    print(f"Results   → {OUTPUT_DIR}/")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
