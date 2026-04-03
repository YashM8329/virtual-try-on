import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Try to import optional dependencies
try:
    from facenet_pytorch import MTCNN
except ImportError:
    print("[FaceEnhancer] Warning: facenet-pytorch not found. Face restoration will be skipped.")
    MTCNN = None

try:
    from gfpgan import GFPGANer
except ImportError:
    print("[FaceEnhancer] Warning: gfpgan not found. Face restoration will be skipped.")
    GFPGANer = None

try:
    from rembg import remove
except ImportError:
    print("[FaceEnhancer] Warning: rembg not found. Bokeh effect will be skipped.")
    remove = None

class FaceEnhancer:
    def __init__(self, device="cpu", model_path="weights/GFPGANv1.4.pth"):
        self.device = device
        self.model_path = model_path
        self._mtcnn = None
        self._gfpgan = None

    def _get_mtcnn(self):
        if self._mtcnn is None and MTCNN is not None:
            self._mtcnn = MTCNN(keep_all=True, device=self.device)
        return self._mtcnn

    def _get_gfpgan(self):
        if self._gfpgan is None and GFPGANer is not None:
            if not os.path.exists(self.model_path):
                print(f"[FaceEnhancer] GFPGAN weights not found at {self.model_path}")
                return None
            
            self._gfpgan = GFPGANer(
                model_path=self.model_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device,
            )
        return self._gfpgan

    def apply_clahe(self, img_cv2, clip=1.1):
        lab = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def unsharp_mask(self, img_cv2, amount=0.5, radius=1):
        blurred = cv2.GaussianBlur(img_cv2, (0,0), sigmaX=radius)
        return cv2.addWeighted(img_cv2, 1 + amount, blurred, -amount, 0)

    def denoise(self, img_bgr):
        return cv2.fastNlMeansDenoisingColored(img_bgr, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)

    def restore_faces(self, pil_img, strength=0.6):
        if MTCNN is None or GFPGANer is None:
            return pil_img

        mtcnn = self._get_mtcnn()
        gfpgan = self._get_gfpgan()

        if mtcnn is None or gfpgan is None:
            return pil_img

        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]

        boxes, _ = mtcnn.detect(pil_img)
        if boxes is None:
            return pil_img

        out_img = img_bgr.copy()
        for box in boxes:
            x1, y1, x2, y2 = [int(max(0, v)) for v in box]
            pad = int(0.3 * max(x2 - x1, y2 - y1))

            x1n = max(0, x1 - pad)
            y1n = max(0, y1 - pad)
            x2n = min(w, x2 + pad)
            y2n = min(h, y2 + pad)

            crop = img_bgr[y1n:y2n, x1n:x2n]
            try:
                _, _, restored = gfpgan.enhance(crop, has_aligned=False, only_center_face=False, paste_back=True)
                blended = ((1 - strength) * crop + strength * restored).astype("uint8")
                out_img[y1n:y2n, x1n:x2n] = blended
            except Exception as e:
                print(f"[FaceEnhancer] face_restore warning: {e}")

        return Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))

    def get_foreground_mask(self, pil_img):
        if remove is None:
            return None
        
        bio = BytesIO()
        pil_img.save(bio, format="PNG")
        in_bytes = bio.getvalue()
        out = remove(in_bytes)
        out_pil = Image.open(BytesIO(out)).convert("RGBA")
        alpha = out_pil.split()[-1]
        alpha_np = np.array(alpha).astype(np.uint8)
        _, mask = cv2.threshold(alpha_np, 10, 255, cv2.THRESH_BINARY)
        return mask

    def apply_bokeh(self, cv2_img, mask_np, blur_strength=15.0):
        mask = mask_np.astype(np.float32) / 255.0
        feather_size = 5
        mask = cv2.GaussianBlur(mask, (feather_size, feather_size), 0)
        mask3 = cv2.merge([mask, mask, mask])
        
        ksize = int(max(1, blur_strength) * 2 + 1)
        bg = cv2.GaussianBlur(cv2_img, (ksize, ksize), 0)
        
        fg = (cv2_img.astype("float32") * mask3 + bg.astype("float32") * (1 - mask3))
        return fg.clip(0, 255).astype("uint8")

    def enhance_full(self, pil_img, bokeh_strength=15.0, face_restore_strength=0.6, save_name=None, save_dir="enhanced"):
        # Step 1: Denoise
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img_db = self.denoise(img_cv)

        # Step 2: Color correction
        img_cc = self.apply_clahe(img_db, clip=1.1)

        # Step 3: Face restoration
        img_pil_cc = Image.fromarray(cv2.cvtColor(img_cc, cv2.COLOR_BGR2RGB))
        img_face_pil = self.restore_faces(img_pil_cc, strength=face_restore_strength)
        
        # Step 4: Bokeh (Optional based on rembg availability)
        if remove is not None and bokeh_strength > 0:
            mask_np = self.get_foreground_mask(img_face_pil)
            if mask_np is not None:
                img_cv_face = cv2.cvtColor(np.array(img_face_pil), cv2.COLOR_RGB2BGR)
                img_bokeh = self.apply_bokeh(img_cv_face, mask_np, blur_strength=bokeh_strength)
                img_face_pil = Image.fromarray(cv2.cvtColor(img_bokeh, cv2.COLOR_BGR2RGB))

        # Step 5: Final sharpening
        img_cv_final = cv2.cvtColor(np.array(img_face_pil), cv2.COLOR_RGB2BGR)
        img_sh = self.unsharp_mask(img_cv_final, amount=0.5, radius=1)
        
        final_pil = Image.fromarray(cv2.cvtColor(img_sh, cv2.COLOR_BGR2RGB))

        # [NEW] Save logic inside the enhancer
        if save_name:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{save_name}_enhanced.png")
            final_pil.save(save_path)
            print(f"      [FaceEnhancer] Saved enhanced image to {save_path}")

        return final_pil
