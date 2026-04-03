"""
High-level pipeline orchestration:
1) Resize
2) Deblur + denoise
3) Face restore (constrained blending)
4) Segmentation -> bokeh background
5) Color grading + unsharp mask
"""

from utils import pil_to_cv2, cv2_to_pil, resize_max
from pipeline.deblur_denoise import deblur_if_needed
from pipeline.segmentation import get_foreground_mask_pil, apply_bokeh
from pipeline.face_restore import restore_faces_pil
import cv2
import numpy as np
from PIL import Image

def unsharp_mask(img_cv2, amount=1.0, radius=1):
    """Simple unsharp mask using gaussian blur"""
    blurred = cv2.GaussianBlur(img_cv2, (0,0), sigmaX=radius)
    return cv2.addWeighted(img_cv2, 1 + amount, blurred, -amount, 0)

# def apply_clahe(img_cv2):
#     lab = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
#     cl = clahe.apply(l)
#     limg = cv2.merge((cl,a,b))
#     final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#     return final

# def enhance_image(pil_img, bokeh_strength=15.0, face_restore_strength=0.8):
#     # Step 0: resize for speed
#     img_cv = pil_to_cv2(pil_img)
#     img_cv = resize_max(img_cv, max_dim=1024)

#     # Step 1: deblur + denoise
#     img_db = deblur_if_needed(img_cv, threshold=110.0)

#     # Step 2: lightweight global color correction
#     img_cc = apply_clahe(img_db)

#     # Step 3: face restore on the RGB PIL image
#     img_pil_cc = cv2_to_pil(img_cc)
#     img_face = restore_faces_pil(img_pil_cc, strength=face_restore_strength, device="cpu")

#     # Step 4: segmentation & bokeh using mask
#     mask_pil = get_foreground_mask_pil(img_face)
#     img_cv_face = pil_to_cv2(img_face)
#     img_bokeh = apply_bokeh(img_cv_face, mask_pil, blur_strength=bokeh_strength)

#     # Step 5: final sharpening and tone
#     img_sh = unsharp_mask(img_bokeh, amount=0.8, radius=1)
#     img_final = apply_clahe(img_sh)

#     # ensure uint8
#     img_final = np.clip(img_final, 0, 255).astype("uint8")
#     return cv2_to_pil(img_final)

def apply_clahe(img_cv2, clip=1.2): # Lowered default from 2.0 to 1.2
    lab = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def enhance_image(pil_img, bokeh_strength=15.0, face_restore_strength=0.6): # Lowered face strength
    img_cv = pil_to_cv2(pil_img)
    img_cv = resize_max(img_cv, max_dim=1024)

    # Step 1: light deblur/denoise
    img_db = deblur_if_needed(img_cv, threshold=110.0)

    # Step 2: Very mild color correction
    img_cc = apply_clahe(img_db, clip=1.1)

    # Step 3: Face restoration
    img_pil_cc = cv2_to_pil(img_cc)
    # We use a lower strength (0.6) to keep it looking natural
    img_face = restore_faces_pil(img_pil_cc, strength=face_restore_strength)

    # Step 4: Segmentation & Bokeh
    mask_pil = get_foreground_mask_pil(img_face)
    img_cv_face = pil_to_cv2(img_face)
    img_bokeh = apply_bokeh(img_cv_face, mask_pil, blur_strength=bokeh_strength)

    # Step 5: Final touch (Subtle sharpening)
    img_final = unsharp_mask(img_bokeh, amount=0.5, radius=1)
    
    return cv2_to_pil(img_final)
