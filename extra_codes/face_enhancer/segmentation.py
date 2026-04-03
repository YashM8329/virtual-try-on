"""
Background segmentation using rembg (U2Net based).
rembg removes background returning a transparent PNG (alpha).
We use that alpha as a mask to composite blurred background.
"""

from rembg import remove
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

def get_foreground_mask_pil(pil_img):
    """
    Returns mask PIL.Image (mode 'L') where 255 = foreground, 0 = background
    rembg returns RGBA with transparent background.
    """
    # rembg expects bytes
    bio = BytesIO()
    pil_img.save(bio, format="PNG")
    in_bytes = bio.getvalue()
    out = remove(in_bytes)  # bytes of PNG RGBA
    out_pil = Image.open(BytesIO(out)).convert("RGBA")
    # extract alpha
    alpha = out_pil.split()[-1]
    # convert alpha to binary mask via threshold
    alpha_np = np.array(alpha).astype(np.uint8)
    _, mask = cv2.threshold(alpha_np, 10, 255, cv2.THRESH_BINARY)
    mask_pil = Image.fromarray(mask)
    return mask_pil

# def apply_bokeh(cv2_img, mask_pil, blur_strength=15.0):
#     """
#     cv2_img: BGR uint8
#     mask_pil: PIL L mask
#     blur_strength: gaussian blur kernel radius
#     """
#     mask = np.array(mask_pil).astype(np.uint8)
#     mask3 = cv2.merge([mask, mask, mask]) / 255.0
#     # blurred background
#     k = int(max(1, blur_strength) // 1)
#     # use large gaussian blur (ensure odd kernel)
#     ksize = int(max(1, blur_strength) * 2 + 1)
#     bg = cv2.GaussianBlur(cv2_img, (ksize, ksize), sigmaX=0)
#     # composite subject over blurred background
#     fg = (cv2_img.astype("float32") * mask3 + bg.astype("float32") * (1 - mask3)).astype("uint8")
#     return fg

# def apply_bokeh(cv2_img, mask_pil, blur_strength=15.0):
#     mask = np.array(mask_pil).astype(np.float32) / 255.0
    
#     # NEW: Soften the mask edges (Feathering)
#     # This prevents the "white halo" effect
#     mask = cv2.GaussianBlur(mask, (21, 21), 0)
    
#     mask3 = cv2.merge([mask, mask, mask])

#     # blurred background
#     ksize = int(blur_strength * 2 + 1)
#     bg = cv2.GaussianBlur(cv2_img, (ksize, ksize), 0)
    
#     # composite using the soft mask
#     fg = (cv2_img.astype("float32") * mask3 + bg.astype("float32") * (1 - mask3))
#     return fg.astype("uint8")

def apply_bokeh(cv2_img, mask_pil, blur_strength=15.0):
    """
    cv2_img: BGR uint8
    mask_pil: PIL L mask
    blur_strength: gaussian blur kernel radius
    """
    # 1. Convert mask to float32 (0.0 to 1.0)
    mask = np.array(mask_pil).astype(np.float32) / 255.0
    
    # 2. FEATHERING: Soften the mask edges
    # This removes the "halo" and makes the hair blend naturally.
    # We use a small blur on the mask itself.
    feather_size = 5  # Try 5, 7, or 9 for different levels of softness
    if feather_size % 2 == 0: feather_size += 1
    mask = cv2.GaussianBlur(mask, (feather_size, feather_size), 0)
    
    # 3. Create the 3-channel mask for blending
    mask3 = cv2.merge([mask, mask, mask])
    
    # 4. Create the blurred background
    ksize = int(max(1, blur_strength) * 2 + 1)
    bg = cv2.GaussianBlur(cv2_img, (ksize, ksize), 0)
    
    # 5. Composite (Math: Image * Mask + Background * (1 - Mask))
    # Using the softened mask creates a gradient transition at the edges
    fg = (cv2_img.astype("float32") * mask3 + bg.astype("float32") * (1 - mask3))
    
    return fg.clip(0, 255).astype("uint8")

