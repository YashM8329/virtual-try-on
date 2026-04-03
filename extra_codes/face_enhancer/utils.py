import numpy as np
import cv2
from PIL import Image

def pil_to_cv2(img: Image.Image):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img_cv2):
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

def resize_max(img_cv2, max_dim=1024):
    h, w = img_cv2.shape[:2]
    scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
    if scale != 1.0:
        img_cv2 = cv2.resize(img_cv2, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_cv2

def normalize_uint8(img_cv2):
    img_cv2 = np.clip(img_cv2, 0, 255).astype("uint8")
    return img_cv2
