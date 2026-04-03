import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from gfpgan import GFPGANer

_mtcnn = None
_gfpgan = None


def get_mtcnn(device="cpu"):
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(keep_all=True, device=device)
    return _mtcnn


def get_gfpgan(device="cpu"):
    global _gfpgan
    if _gfpgan is None:
        model_path = os.path.join("weights", "GFPGANv1.4.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GFPGAN model not found at {model_path}. "
                f"Please place GFPGANv1.4.pth inside the weights/ folder."
            )

        _gfpgan = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )
    return _gfpgan


def restore_faces_pil(pil_img, strength=0.8, device="cpu"):
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    mtcnn = get_mtcnn(device=device)
    gfpgan = get_gfpgan(device=device)

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
            _, _, restored = gfpgan.enhance(crop, has_aligned=False, only_center_face=False, paste_back=True
            )
            blended = ((1 - strength) * crop + strength * restored).astype("uint8")
            out_img[y1n:y2n, x1n:x2n] = blended
        except Exception as e:
            print("[face_restore] warning:", e)

    return Image.fromarray(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
