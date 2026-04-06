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
from PIL import Image

# ── Optional neural net face restoration (used at very low strength) ──────────
try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

try:
    from gfpgan import GFPGANer
except ImportError:
    GFPGANer = None


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

def detect_faces(pil_img: Image.Image):
    """Return list of (x1,y1,x2,y2) face boxes using MTCNN or OpenCV fallback."""
    boxes = []

    # --- MTCNN (better) ---
    if MTCNN is not None:
        try:
            mtcnn = MTCNN(keep_all=True, device="cpu")
            detected, _ = mtcnn.detect(pil_img)
            if detected is not None:
                for b in detected:
                    boxes.append(tuple(int(max(0, v)) for v in b))
                return boxes
        except Exception:
            pass

    # --- OpenCV DNN face detector fallback ---
    img_bgr = _bgr(pil_img)
    h, w = img_bgr.shape[:2]
    # Use Haar cascade as last resort
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces):
        for (x, y, fw, fh) in faces:
            boxes.append((x, y, x + fw, y + fh))

    return boxes


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

    # Subtle warmth: nudge blue channel slightly down, red slightly up
    if warmth > 0:
        b_ch, g_ch, r_ch = cv2.split(result)
        r_ch = np.clip(r_ch * (1 + warmth * m), 0, 255)
        b_ch = np.clip(b_ch * (1 - warmth * 0.5 * m), 0, 255)
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
#  7. Light denoise (NL-means at very low h)
# ─────────────────────────────────────────────────────────────────
def light_denoise(img_bgr: np.ndarray) -> np.ndarray:
    # h=1 — minimal noise removal; preserves fine beard/hair strands and pore detail
    return cv2.fastNlMeansDenoisingColored(
        img_bgr, None, h=1, hColor=1, templateWindowSize=7, searchWindowSize=21
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Main class — drop-in replacement for the old FaceEnhancer
# ═════════════════════════════════════════════════════════════════════════════

class FaceEnhancer:
    """
    Studio-quality portrait enhancer.

    Pipeline (all steps work without neural nets):
      1. Light denoise  (h=2 — removes grain, keeps pores)
      2. Frequency-separation skin polish  (bilateral LF smoothing)
      3. CLAHE global contrast normalisation
      4. Studio face glow  (screen-dodge LAB lift on face region)
      5. Micro-sharpen on face  (very small radius — just features)
      6. Studio colour grade  (S-curve + fill lift)
      7. Subtle vignette
      [Optional] GFPGAN at very low blend (≤0.25) if weights available
    """

    def __init__(self, device: str = "cpu", model_path: str = "weights/GFPGANv1.4.pth"):
        self.device = device
        self.model_path = model_path
        self._mtcnn = None
        self._gfpgan = None

    # ── lazy GFPGAN (only if weights present) ────────────────────────────────
    def _get_gfpgan(self):
        if self._gfpgan is None and GFPGANer is not None:
            if os.path.exists(self.model_path):
                self._gfpgan = GFPGANer(
                    model_path=self.model_path,
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device,
                )
        return self._gfpgan

    # ── optional very-light GFPGAN pass ──────────────────────────────────────
    def _apply_gfpgan_light(self, img_bgr: np.ndarray, strength: float = 0.20) -> np.ndarray:
        """Run GFPGAN at ≤0.25 blend — just to recover compression artefacts."""
        gfpgan = self._get_gfpgan()
        if gfpgan is None:
            return img_bgr
        try:
            _, _, restored = gfpgan.enhance(
                img_bgr, has_aligned=False, only_center_face=False, paste_back=True
            )
            return _u8(_f32(img_bgr) * (1 - strength) + _f32(restored) * strength)
        except Exception as e:
            print(f"[FaceEnhancer] GFPGAN skipped: {e}")
            return img_bgr

    # ── main entry point ──────────────────────────────────────────────────────
    def enhance_full(
        self,
        pil_img: Image.Image,
        # Legacy kwarg kept for back-compat — ignored
        bokeh_strength: float = 0,
        face_restore_strength: float = 0.20,   # was 0.6 — now much lighter
        save_name: str = None,
        save_dir: str = "enhanced",
    ) -> Image.Image:
        """
        Full studio-quality enhancement pipeline.

        face_restore_strength  ← kept at 0.20 max to avoid GFPGAN blurring.
                                  If GFPGAN weights are missing the step is
                                  skipped gracefully.
        """
        img_bgr = _bgr(pil_img)

        # ── Step 1: Light denoise ─────────────────────────────────────────
        img_bgr = light_denoise(img_bgr)

        # ── Step 2: Detect faces + build soft mask ────────────────────────
        boxes = detect_faces(pil_img)
        h, w = img_bgr.shape[:2]
        if boxes:
            face_mask = make_face_mask((h, w), boxes, pad_ratio=0.30, feather=0.14)
        else:
            # Fallback: assume central 60% of image is face
            face_mask = np.zeros((h, w), dtype=np.float32)
            y0, y1 = int(h * 0.1), int(h * 0.9)
            x0, x1 = int(w * 0.15), int(w * 0.85)
            face_mask[y0:y1, x0:x1] = 1.0
            ksize = (h // 6) | 1
            face_mask = cv2.GaussianBlur(face_mask, (ksize, ksize), ksize * 0.3)

        # ── Step 3: Frequency-separation skin smooth (face region only) ──
        img_smooth = freq_sep_smooth(img_bgr, radius=8, strength=0.28)  # ← was 0.52; lower = more natural texture/beard detail preserved
        # Blend smooth only where face mask is active
        m3 = face_mask[:, :, np.newaxis]
        img_bgr = _u8(_f32(img_bgr) * (1 - m3) + _f32(img_smooth) * m3)

        # ── Step 4: CLAHE global contrast ────────────────────────────────
        img_bgr = apply_clahe(img_bgr, clip=1.2)  # ← was 1.4; reduced to avoid contrast flattening

        # ── Step 5: Studio face glow ──────────────────────────────────────
        img_bgr = studio_face_glow(img_bgr, face_mask, glow_strength=0.10, warmth=0.02)  # ← was 0.26/0.035; reduced to preserve shadow depth

        # ── Step 6: Micro-sharpen on face (crisp features) ───────────────
        # Pass 1 — tight-radius USM for fine edge crispness (eyes, brow hairs, beard)
        img_sharp = micro_sharpen(img_bgr, amount=0.85, radius=0.5)  # ← raised from 0.55; smaller radius = only lifts thin edges
        img_bgr = _u8(_f32(img_bgr) * (1 - m3) + _f32(img_sharp) * m3)

        # Pass 2 — medium-radius USM for facial structure (nose bridge, lip contour, jaw)
        img_sharp2 = micro_sharpen(img_bgr, amount=0.40, radius=1.2)
        img_bgr = _u8(_f32(img_bgr) * (1 - m3 * 0.6) + _f32(img_sharp2) * m3 * 0.6)

        # ── Step 7 (optional): Very light GFPGAN ─────────────────────────
        if face_restore_strength > 0:
            img_bgr = self._apply_gfpgan_light(img_bgr, strength=min(face_restore_strength, 0.25))

        # ── Step 8: Studio colour grade ───────────────────────────────────
        img_bgr = studio_color_grade(img_bgr)

        # ── Step 9: Subtle vignette ───────────────────────────────────────
        img_bgr = add_vignette(img_bgr, strength=0.20)

        final_pil = _pil(img_bgr)

        if save_name:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{save_name}_enhanced.png")
            final_pil.save(save_path)
            print(f"[FaceEnhancer] Saved → {save_path}")

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