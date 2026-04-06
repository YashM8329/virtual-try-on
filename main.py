"""
main.py
Entry point for the Virtual Try-On pipeline.

Key changes vs previous version
────────────────────────────────
1. Two-pass loop: pre-compute ALL masks first, then run ALL diffusion.
   Keeps GPU warm between diffusion calls; eliminates cold-start on
   each image.

2. Face enhancement bokeh disabled (bokeh_strength=0).
   rembg (U2Net) was running a full background-removal neural net
   per image just for background blur — expensive with no quality
   gain on the hoodie output.

3. Mask debug save moved to a background thread (ThreadPoolExecutor).
   matplotlib savefig at 18×6 figsize was blocking the CPU for
   ~1-2 s per image while the GPU sat idle.

4. image_path forwarded to run_inpainting for per-image seeding.

5. SAM predictor is imported but currently unused in the clothing-mask
   path.  It is still loaded so the pipeline stays drop-in compatible.
"""

import os
import glob
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

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
from image_utils      import preprocess_image, save_images
from pose_extraction  import extract_pose_landmarks, get_torso_bbox
from skin_segmentation import get_skin_mask
from clothing_mask    import get_clothing_mask
from inpainting       import run_inpainting
from neck_blend       import apply_neck_blend
# from autocrop         import crop_to_hands
from scorecard_processor import ScorecardProcessor

# ── Step 1: Load all models once ──────────────────────────────────────────────
from model_loader import (
    load_sam, load_pose_landmarker, load_skin_segmenter,
    load_diffusion_pipeline, load_openpose_generator,
)

print("\n[Model Loading] Loading all models...")
sam_predictor  = load_sam()
pose_detector  = load_pose_landmarker()
skin_segmenter = load_skin_segmenter()
openpose_gen   = load_openpose_generator()
pipe           = load_diffusion_pipeline()
face_enhancer  = FaceEnhancer()
print("[Model Loading] All models loaded.\n")

# ── Scorecard processor ────────────────────────────────────────────────────────
print("[Scorecard] Initializing post-processing models...")
scorecard_proc = ScorecardProcessor(
    "weights/modnet_photographic_portrait_matting.onnx",
    "template/template_1/template_1696_2528.png",
    "template/template_1/template_not_human_1696_2528.png",
)

# ── Background thread pool for non-blocking mask debug saves ──────────────────
_save_pool = ThreadPoolExecutor(max_workers=2)


def _save_mask_debug_async(img_rgb, refined_mask, soft_mask, save_path):
    """Save mask debug composite in a background thread (non-blocking)."""
    grey_overlay = img_rgb.copy().astype(np.float32)
    alpha_arr    = (refined_mask.astype(np.float32) / 255.0)[..., np.newaxis]
    grey_color   = np.array([128, 128, 128], dtype=np.float32)
    grey_overlay = (grey_color * alpha_arr + grey_overlay * (1 - alpha_arr)).astype(np.uint8)
    save_images(
        {
            "Original":                  img_rgb,
            "Refined Binary Mask":       refined_mask,
            "Soft Mask (Gaussian)":      soft_mask,
            "Mask Overlay (IDM style)":  grey_overlay,
        },
        save_path=save_path,
        figsize=(18, 6),
    )


# ══════════════════════════════════════════════════════════════════════════════
# PASS 1 — Pre-compute all masks (CPU-heavy, GPU idle)
# ══════════════════════════════════════════════════════════════════════════════

def precompute_single(image_path: str):
    """
    Run all CPU-side preprocessing for one image and return everything
    needed by the diffusion step.

    Returns dict with keys: img_pil, soft_mask, fname
    Returns None on failure.
    """
    fname = os.path.splitext(os.path.basename(image_path))[0]
    try:
        # Step 1 — Load + resize
        img_pil, img_bgr, (h, w) = preprocess_image(image_path, target_size=768)
        img_rgb = np.array(img_pil)
        print(f"  [precompute] {fname}: loaded {w}×{h}")

        # Step 2 — Remove background and replace with White
        # This gives the AI a clean slate for hoodie generation
        bgra = scorecard_proc.remove_background(img_bgr)
        alpha = bgra[:, :, 3] / 255.0
        img_bgr = (bgra[:, :, :3] * alpha[:, :, np.newaxis] + (1 - alpha[:, :, np.newaxis]) * 255).astype(np.uint8)
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        img_rgb = np.array(img_pil)

        # Step 3 — Pose landmarks
        lm_px, _ = extract_pose_landmarks(img_rgb, pose_detector)
        torso_bbox = get_torso_bbox(lm_px, h, w, pad_v=0.08, pad_h=0.18)

        # Step 4 — Horizontal crop to hands (Commented out)
        # img_pil, img_bgr, (h, w) = crop_to_hands(img_pil, img_bgr, lm_px, fname=fname)
        # img_rgb = np.array(img_pil)

        # Step 5 — (Face enhancement moved to end of pipeline)

        # Step 3 — Skin mask
        skin_mask = get_skin_mask(img_rgb, skin_segmenter)

        # Step 4 — Clothing mask
        refined_mask, soft_mask = get_clothing_mask(img_rgb, lm_px, skin_mask, torso_bbox)
        soft_mask = apply_neck_blend(soft_mask, skin_mask)

        # Save mask debug asynchronously (does not block the loop)
        mask_path = os.path.join(MASK_DIR, f"{fname}_mask_debug.png")
        _save_pool.submit(_save_mask_debug_async, img_rgb, refined_mask, soft_mask, mask_path)
        print(f"  [precompute] {fname}: mask queued for async save")

        return {"img_pil": img_pil, "soft_mask": soft_mask, "fname": fname,
                "image_path": image_path}

    except Exception as e:
        print(f"  [precompute] ✗ {fname}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PASS 2 — Diffusion (GPU-heavy)
# ══════════════════════════════════════════════════════════════════════════════

def diffuse_single(item: dict):
    """
    Run ControlNet inpainting for one pre-computed item.
    Returns (result_pil, fname) or raises on failure.
    """
    return run_inpainting(
        img_pil=item["img_pil"],
        soft_mask=item["soft_mask"],
        pipe=pipe,
        openpose_gen=openpose_gen,
        image_path=item["image_path"],
    ), item["fname"]


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

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

    # ── Pass 1: pre-compute masks for all images ───────────────────────────────
    print("\n── Pass 1: Pre-computing masks ──────────────────────────────────────")
    prepared = []
    for image_path in image_paths:
        item = precompute_single(image_path)
        if item is not None:
            prepared.append(item)

    print(f"\n── Pass 2: Diffusion ({len(prepared)} images) ───────────────────────")

    success_count = 0
    for i, item in enumerate(prepared):
        fname    = item["fname"]
        out_path = os.path.join(OUTPUT_DIR, f"{fname}_green_hoodie.png")

        try:
            result, _ = diffuse_single(item)
            result.save(out_path)
            print(f"[{i+1}/{len(prepared)}] ✓ Try-on saved → {out_path}")

            # ── Step 3: Remove background from generated result ────────────────
            result_bgr  = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            transparent = scorecard_proc.remove_background(result_bgr)

            # ── Step 4: Face enhancement (Final Polish) ───────────────────────
            # Passing the transparent image to enhancer to fix any AI blurring
            img_pil_final = Image.fromarray(cv2.cvtColor(transparent, cv2.COLOR_BGRA2RGBA))
            enhanced_pil  = face_enhancer.enhance_full(img_pil_final, save_name=fname)
            enhanced_bgr  = cv2.cvtColor(np.array(enhanced_pil), cv2.COLOR_RGB2BGR)
            
            # Re-apply alpha channel from transparent result to enhanced result
            final_bgra = np.concatenate([enhanced_bgr, transparent[:, :, 3:4]], axis=2)

            # ── Step 5: Scorecard template overlay ────────────────────────────
            print(f"[{i+1}/{len(prepared)}] Creating scorecard...")
            final_scorecard = scorecard_proc.overlay_on_template(final_bgra)

            if final_scorecard is not None:
                sc_path = os.path.join(SCORECARD_DIR, f"{fname}_scorecard.png")
                cv2.imwrite(sc_path, final_scorecard)
                print(f"[{i+1}/{len(prepared)}] ✓ Scorecard saved → {sc_path}")

            success_count += 1

        except Exception as e:
            print(f"[{i+1}/{len(prepared)}] ✗ Failed — {fname}: {e}")

    # Wait for any pending mask-debug saves before exiting
    _save_pool.shutdown(wait=True)

    print(f"\n{'='*55}")
    print(f"Completed : {success_count}/{len(prepared)} successful")
    print(f"Results   → {OUTPUT_DIR}/")
    print(f"Masks     → {MASK_DIR}/")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()