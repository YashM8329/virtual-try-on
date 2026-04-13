"""
face_enhancer.py  — Studio-quality face enhancement pipeline
──────────────────────────────────────────────────────────────
Goal: make portraits look like studio photos without altering
facial features. Achieves:

  1. Zero blur on face (frequency-separation-based smoothing only on skin)
  2. Minimal sharpening (micro-detail layer, not aggressive USM)
  3. Crisp facial features (eyes, lips, brows enhanced via structure layer)
  4. Natural studio glow (dodge-light on face, not flat brightness boost)
  5. Overall studio-grade finish (colour grade, subtle vignette)

No GFPGAN / heavy neural net required — works purely with classical CV.
If GFPGAN is present it is used at a very low blend so it never blurs.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# ── Optional neural net face restoration (used at very low strength) ──────────
try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

try:
    from facexlib.utils.face_restoration_helper import FaceRestorationHelper
except ImportError:
    FaceRestorationHelper = None


# ═════════════════════════════════════════════════════════════════════════════
#  Utility helpers
# ═════════════════════════════════════════════════════════════════════════════

def _bgr(pil_img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _f32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def _u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
#  Core processing blocks
# ═════════════════════════════════════════════════════════════════════════════


def make_face_mask(shape, boxes, pad_ratio=0.35, feather=0.12):
    """
    Soft elliptical mask covering face region.
    pad_ratio  — how much to expand each bounding box
    feather    — how wide the gaussian falloff is (as fraction of face size)
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    for (x1, y1, x2, y2) in boxes:
        fw, fh = x2 - x1, y2 - y1
        pad_x = int(pad_ratio * fw)
        pad_y = int(pad_ratio * fh)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        rx = (fw // 2) + pad_x
        ry = (fh // 2) + pad_y

        # Draw filled ellipse
        cv2.ellipse(
            mask,
            (cx, cy),
            (rx, ry),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=1.0,
            thickness=-1,
        )

    # Gaussian feather
    blur_r = int(feather * max(h, w))
    blur_r = blur_r | 1  # must be odd
    mask = cv2.GaussianBlur(mask, (blur_r, blur_r), blur_r * 0.33)
    mask = np.clip(mask, 0, 1)
    return mask  # shape (H,W) float32 [0..1]


# ─────────────────────────────────────────────────────────────────
#  1. Frequency-separation skin smoothing
#     High-freq (detail) layer is kept; only low-freq is smoothed.
#     This removes blotchiness WITHOUT blurring pores/hair/eyes.
# ─────────────────────────────────────────────────────────────────
def freq_sep_smooth(img_bgr: np.ndarray, radius: int = 9, strength: float = 0.55) -> np.ndarray:
    """
    Bilateral-filtered low frequency layer blended back at `strength`.
    radius   — spatial neighbourhood (pixels). Keep ≤ 12 for realism.
    strength — 0 = no smoothing, 1 = full bilateral (too heavy).
               0.45-0.65 gives a natural skin polish.
    """
    # Bilateral preserves edges while smoothing flat-skin areas
    smooth = cv2.bilateralFilter(img_bgr, d=radius * 2 + 1,
                                  sigmaColor=30, sigmaSpace=radius)
    # Low-freq layer
    low_freq = cv2.GaussianBlur(img_bgr, (radius * 2 + 1, radius * 2 + 1), 0)
    smooth_low = cv2.GaussianBlur(smooth, (radius * 2 + 1, radius * 2 + 1), 0)

    # High-freq detail = original − low_freq  (add 128 to keep positive)
    detail = _f32(img_bgr) - _f32(low_freq) + 128.0

    # Replace low freq with smooth version, reattach original detail
    result = _f32(smooth_low) * strength + _f32(low_freq) * (1 - strength)
    result = result + _f32(img_bgr) - _f32(low_freq)   # restore all detail
    return _u8(result)


# ─────────────────────────────────────────────────────────────────
#  2. Gentle structure sharpening (face features only)
#     Uses a small-radius unsharp mask to lift micro-contrast on
#     eyes, lips and brows — NOT the wide-radius aggressive USM.
# ─────────────────────────────────────────────────────────────────
def micro_sharpen(img_bgr: np.ndarray, amount: float = 0.30, radius: float = 0.7) -> np.ndarray:
    """
    amount  — 0.2-0.4 is natural; >0.6 starts looking plastic
    radius  — keep small (0.5-1.0 px) — only lifts fine edges
    """
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=radius)
    sharp = cv2.addWeighted(img_bgr, 1 + amount, blurred, -amount, 0)
    return _u8(sharp)


# ─────────────────────────────────────────────────────────────────
#  3. Studio face glow — dodge light on face
#     Creates a soft radial luminosity lift centred on the face.
#     Preserves skin-tone hue (operates in LAB L channel only).
#     Not a flat brightness: brighter midtones, protected shadows.
# ─────────────────────────────────────────────────────────────────
def studio_face_glow(
    img_bgr: np.ndarray,
    face_mask: np.ndarray,         # float32 [0..1], same HW as img
    glow_strength: float = 0.28,   # 0.18-0.35 looks natural
    warmth: float = 0.04,          # slight warm tint (0 = neutral)
) -> np.ndarray:
    """
    Lifts the luminance of the face region via a screen-blend dodge.
    warmth adds a tiny yellow-red shift for that soft studio-key-light feel.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    # Screen-blend lift: result = 1 - (1-L/255)*(1-glow)  scaled to [0,255]
    Ln = L / 255.0
    screen = 1.0 - (1.0 - Ln) * (1.0 - glow_strength)
    L_lifted = screen * 255.0

    # Blend with face mask — only lift inside face ellipse
    m = face_mask  # (H,W) [0..1]
    L_out = L * (1 - m) + L_lifted * m
    L_out = np.clip(L_out, 0, 255)

    # Reconstruct
    lab_out = cv2.merge([L_out, A, B]).astype(np.uint8)
    result = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR).astype(np.float32)

    # Subtle warmth/cool correction: operates on face mask region only
    if warmth != 0:
        b_ch, g_ch, r_ch = cv2.split(result)
        if warmth > 0:
            # Warm shift: red up, blue down
            r_ch = np.clip(r_ch * (1 + warmth * m), 0, 255)
            b_ch = np.clip(b_ch * (1 - warmth * 0.5 * m), 0, 255)
        else:
            # Cool shift: blue up, red down (neutralizes yellow cast)
            cool = abs(warmth)
            b_ch = np.clip(b_ch * (1 + cool * 0.6 * m), 0, 255)
            r_ch = np.clip(r_ch * (1 - cool * 0.4 * m), 0, 255)
        result = cv2.merge([b_ch, g_ch, r_ch])

    return _u8(result)


# ─────────────────────────────────────────────────────────────────
#  4. CLAHE on whole image — gentle contrast normalisation
# ─────────────────────────────────────────────────────────────────
def apply_clahe(img_bgr: np.ndarray, clip: float = 1.4, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────
#  5. Colour grade — studio "clean" look
#     Slight lift of blacks (fill light), highlight rolloff,
#     and a gentle S-curve for depth.
# ─────────────────────────────────────────────────────────────────
def studio_color_grade(img_bgr: np.ndarray) -> np.ndarray:
    """Applies a gentle S-curve + fill-light lift for a studio feel."""
    # Build an S-curve LUT
    x = np.arange(256, dtype=np.float32)

    # Lift shadows slightly (fill light) — blacks → 8
    # Roll off highlights gently — whites → 248
    # Midtone contrast boost via sine
    lift = 3.0  # ← was 8.0; lower lift preserves natural shadow depth (beard, eye sockets)
    ceil = 248.0
    x_norm = x / 255.0
    # S-curve via smooth sigmoid deviation
    s = x_norm + 0.08 * np.sin(np.pi * x_norm)  # gentle S
    # Remap to [lift, ceil]
    lut = (s * (ceil - lift) + lift).clip(0, 255).astype(np.uint8)

    # Apply per-channel
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([cv2.LUT(b, lut), cv2.LUT(g, lut), cv2.LUT(r, lut)])


# ─────────────────────────────────────────────────────────────────
#  6. Subtle lens vignette (darkens corners slightly)
# ─────────────────────────────────────────────────────────────────
def add_vignette(img_bgr: np.ndarray, strength: float = 0.25) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    # Gaussian radial mask centred on image
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    dist = np.clip(dist, 0, 1)
    vignette = 1.0 - strength * dist ** 1.8
    vignette = vignette[:, :, np.newaxis]
    return _u8(_f32(img_bgr) * vignette)



# ─────────────────────────────────────────────────────────────────
#  7. Targeted skin brightening in LAB L-channel
#     Screen-blend lift on skin regions only, with shadow protection
#     so beard/eye-socket depth is preserved.
# ─────────────────────────────────────────────────────────────────
def _skin_brighten_lab(img_bgr: np.ndarray, skin_mask_f: np.ndarray, lift: float = 0.055) -> np.ndarray:
    """
    Lift midtone luminance on skin regions using a screen-blend in LAB L-channel.
    lift=0.055 raises a midtone pixel (L=128) by ~13 L-units (~5% relative).

    Shadow protection: pixels below L≈102 receive progressively less lift,
    preserving beard shadow and eye-socket depth.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    Ln = L / 255.0
    L_lifted = (1.0 - (1.0 - Ln) * (1.0 - lift)) * 255.0
    # Ramp: 0→1 over L=[0,102]; full lift above L=102
    shadow_protect = np.clip(Ln * 2.5, 0, 1)
    blend_weight = skin_mask_f * shadow_protect
    L_out = np.clip(L * (1.0 - blend_weight) + L_lifted * blend_weight, 0, 255)
    lab_out = cv2.merge([L_out, A, B]).astype(np.uint8)
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


# ═════════════════════════════════════════════════════════════════════════════
#  Main class — drop-in replacement for the old FaceEnhancer
# ═════════════════════════════════════════════════════════════════════════════

class FaceEnhancer:
    """
    Studio-quality portrait enhancer.
    
    Pipeline:
      1. AI Face Restoration (CodeFormer ONNX) — Synthesizes realistic skin/details
      2. Frequency-separation skin polish
      3. CLAHE global contrast normalisation
      4. Studio face glow
      5. Micro-sharpen on face
      6. Studio colour grade
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        # Cache MTCNN for fallback detection
        if MTCNN is not None:
            self._mtcnn = MTCNN(keep_all=True, device="cpu") # Keep on CPU to save VRAM
            print("[FaceEnhancer] MTCNN cached.")
        else:
            self._mtcnn = None

        # Load CodeFormer ONNX Session
        self.codeformer_path = "weights/codeformer.onnx"
        if os.path.exists(self.codeformer_path):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
            try:
                self._codeformer_sess = ort.InferenceSession(self.codeformer_path, providers=providers)
                print(f"[FaceEnhancer] CodeFormer ONNX loaded on {device}.")
            except Exception as e:
                print(f"⚠️ Failed to load CodeFormer ONNX: {e}")
                self._codeformer_sess = None
        else:
            print("⚠️ CodeFormer ONNX weights not found at weights/codeformer.onnx. Skipping AI enhancement.")
            self._codeformer_sess = None

    def _run_codeformer(self, img_bgr: np.ndarray, boxes: list) -> np.ndarray:
        """Run CodeFormer ONNX on detected faces."""
        if self._codeformer_sess is None or not boxes:
            return img_bgr

        # Use facexlib helper for alignment math if available
        if FaceRestorationHelper is not None:
            face_helper = FaceRestorationHelper(
                upscale_factor=1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50', # unused for align
                device='cpu'
            )
            face_helper.clean_all()
            face_helper.read_image(img_bgr)
            
            # Since we have boxes, we can manually set them
            # facexlib expects [x1, y1, x2, y2, score]
            face_helper.face_det_output = [list(b) + [1.0] for b in boxes]
            face_helper.get_face_landmarks_5(only_center_face=False)
            face_helper.align_warp_face()

            for i, cropped_face in enumerate(face_helper.cropped_faces):
                # Preprocess: RGB, [0, 1], normalize to [-1, 1], (1, 3, 512, 512)
                face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                face_input = face_rgb.astype(np.float32) / 255.0
                face_input = (face_input - 0.5) / 0.5
                face_input = np.transpose(face_input, (2, 0, 1))[np.newaxis, :]

                # Inference
                # CodeFormer expects two inputs: 'input' (image) and 'weight' (fidelity balance)
                # weight=0.5 is the standard balance between restoration and original identity.
                ort_inputs = {
                    self._codeformer_sess.get_inputs()[0].name: face_input,
                    "weight": np.array([0.5], dtype=np.float64)
                }
                output = self._codeformer_sess.run(None, ort_inputs)[0][0]

                # Postprocess: (3, 512, 512) -> (512, 512, 3), RGB -> BGR
                output = np.transpose(output, (1, 2, 0))
                output = (output * 0.5 + 0.5).clip(0, 1) * 255.0
                restored_face = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)
                face_helper.add_restored_face(restored_face)

            face_helper.get_inverse_affine(None)
            restored_img = face_helper.paste_faces_to_input_image()
            return restored_img
        else:
            # Simple crop fallback if facexlib missing
            restored_img = img_bgr.copy()
            for (x1, y1, x2, y2) in boxes:
                fw, fh = x2 - x1, y2 - y1
                # Square crop
                side = max(fw, fh)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                x1_s, y1_s = max(0, cx - side // 2), max(0, cy - side // 2)
                x2_s, y2_s = min(img_bgr.shape[1], x1_s + side), min(img_bgr.shape[0], y1_s + side)
                
                face_crop = img_bgr[y1_s:y2_s, x1_s:x2_s]
                face_resized = cv2.resize(face_crop, (512, 512))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_input = face_rgb.astype(np.float32) / 255.0
                face_input = (face_input - 0.5) / 0.5
                face_input = np.transpose(face_input, (2, 0, 1))[np.newaxis, :]

                ort_inputs = {
                    self._codeformer_sess.get_inputs()[0].name: face_input,
                    "weight": np.array([0.5], dtype=np.float64)
                }
                output = self._codeformer_sess.run(None, ort_inputs)[0][0]
                
                output = np.transpose(output, (1, 2, 0))
                output = (output * 0.5 + 0.5).clip(0, 1) * 255.0
                restored_face = cv2.resize(cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR), (x2_s - x1_s, y2_s - y1_s))
                restored_img[y1_s:y2_s, x1_s:x2_s] = restored_face
            return restored_img

    # ── main entry point ──────────────────────────────────────────────────────
    def enhance_full(
        self,
        pil_img: Image.Image,
        save_name: str = None,
        save_dir: str = "enhanced",
        user_landmarks: dict = None,
        skin_mask: np.ndarray = None,   # (H, W) uint8 0-255 from MediaPipe segmentation
    ) -> Image.Image:
        img_bgr = _bgr(pil_img)
        # Faster alternative to light_denoise
        img_bgr = cv2.bilateralFilter(img_bgr, d=5, sigmaColor=15, sigmaSpace=15)

        # Normalize skin_mask to float32 [0..1] for blending operations
        if skin_mask is not None:
            skin_mask_f = skin_mask.astype(np.float32) / 255.0   # (H, W) [0..1]
            skin_mask_3 = skin_mask_f[:, :, np.newaxis]           # (H, W, 1) for broadcasting
        else:
            skin_mask_f = skin_mask_3 = None

        # ── Step 2: Detect faces + build soft mask ────────────────────────
        h, w = img_bgr.shape[:2]
        boxes = []
        
        # USE MEDIA PIPE LANDMARKS IF PROVIDED (Saves ~1-2s)
        if user_landmarks:
            # Face landmarks: 0 (nose), 1-6 (eyes), 7-10 (ears)
            # We can use eye landmarks 1, 4 and nose 0 to get a box
            try:
                nose = user_landmarks.get(0)
                l_eye = user_landmarks.get(1)
                r_eye = user_landmarks.get(4)
                if nose and l_eye and r_eye:
                    eye_dist = abs(r_eye[0] - l_eye[0])
                    x1 = max(0, int(min(l_eye[0], r_eye[0]) - eye_dist * 0.8))
                    x2 = min(w, int(max(l_eye[0], r_eye[0]) + eye_dist * 0.8))
                    y1 = max(0, int(min(l_eye[1], r_eye[1]) - eye_dist * 1.2))
                    y2 = min(h, int(nose[1] + eye_dist * 1.2))
                    boxes = [(x1, y1, x2, y2)]
            except: pass

        if not boxes and self._mtcnn:
            try:
                detected, _ = self._mtcnn.detect(pil_img)
                if detected is not None:
                    boxes = [tuple(int(max(0, v)) for v in b) for b in detected]
            except: pass
            
        if boxes:
            face_mask = make_face_mask((h, w), boxes, pad_ratio=0.30, feather=0.14)
        else:
            face_mask = np.zeros((h, w), dtype=np.float32)
            face_mask[int(h*0.1):int(h*0.9), int(w*0.15):int(w*0.85)] = 1.0
            k = (h // 6) | 1
            face_mask = cv2.GaussianBlur(face_mask, (k, k), k * 0.3)

        m3 = face_mask[:, :, np.newaxis]

        # ── Step 2: AI Face Restoration (CodeFormer) ──────────────────────
        ai_restored = False
        if self._codeformer_sess is not None:
            img_bgr = self._run_codeformer(img_bgr, boxes)
            ai_restored = True

        # ── Step 3: Freq-sep skin smoothing (always — post-CodeFormer at reduced strength) ──
        _smooth_strength = 0.20 if ai_restored else 0.28
        _smooth_radius   = 7    if ai_restored else 8
        img_smooth = freq_sep_smooth(img_bgr, radius=_smooth_radius, strength=_smooth_strength)
        # Use precise skin_mask when available; fall back to face ellipse
        _smooth_region = skin_mask_3 if skin_mask_3 is not None else m3
        img_bgr = _u8(_f32(img_bgr) * (1 - _smooth_region) + _f32(img_smooth) * _smooth_region)

        # ── Step 4: CLAHE global contrast normalisation ───────────────────
        img_bgr = apply_clahe(img_bgr, clip=1.6)

        # ── Step 4b: Targeted skin brightening (LAB midtone lift, shadow-safe) ──
        if skin_mask_f is not None:
            img_bgr = _skin_brighten_lab(img_bgr, skin_mask_f, lift=0.055)

        # ── Step 5: Studio face glow — increased strength + cool correction ──
        img_bgr = studio_face_glow(img_bgr, face_mask, glow_strength=0.22, warmth=-0.015)

        # ── Step 6: Two-pass sharpening ───────────────────────────────────
        # Pass 1: skin texture (reduced — CodeFormer already restored detail)
        img_sharp = micro_sharpen(img_bgr, amount=0.30 if ai_restored else 0.70, radius=0.5)
        img_bgr = _u8(_f32(img_bgr) * (1 - m3) + _f32(img_sharp) * m3)
        # Pass 2: feature edges — eyes, brows, beard (tighter radius for crisp structure)
        img_edge_sharp = micro_sharpen(img_bgr, amount=0.55, radius=0.4)
        img_bgr = _u8(_f32(img_bgr) * (1 - m3) + _f32(img_edge_sharp) * m3)

        img_bgr = studio_color_grade(img_bgr)
        img_bgr = add_vignette(img_bgr, strength=0.20)
        
        final_pil = _pil(img_bgr)

        if save_name:
            os.makedirs(save_dir, exist_ok=True)
            final_pil.save(os.path.join(save_dir, f"{save_name}_enhanced.png"))
        return final_pil


# ═════════════════════════════════════════════════════════════════════════════
#  Quick test  (run:  python face_enhancer.py input.jpg)
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python face_enhancer.py <input_image> [output_image]")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else "enhanced_output.png"

    print(f"Enhancing {src} …")
    img = Image.open(src).convert("RGB")
    fe = FaceEnhancer()
    out = fe.enhance_full(img, save_name=None)
    out.save(dst)
    print(f"Saved → {dst}")