"""
Simple deblur + denoise utilities.
We use:
- variance-of-laplacian blur detection
- light Richardson-Lucy (skimage) deconvolution when blur is detected
- OpenCV fastNlMeansDenoisingColored for denoising
"""

import cv2
import numpy as np
from skimage.restoration import richardson_lucy
from skimage.color import rgb2gray
from skimage import img_as_float
from utils import normalize_uint8

def blur_metric_gray(img_gray):
    # variance of Laplacian
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def estimate_motion_psf(length=15, angle=0):
    """
    Simple linear motion PSF kernel (approximate).
    length: pixels
    angle: degrees
    """
    size = length if length % 2 == 1 else length + 1
    psf = np.zeros((size, size))
    center = size // 2
    angle_rad = np.deg2rad(angle)
    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)
    for i in range(size):
        x = i - center
        y = int(round(center + x * np.tan(angle_rad)))
        if 0 <= y < size:
            psf[y, i] = 1
    psf = psf / psf.sum() if psf.sum() != 0 else np.ones_like(psf) / psf.size
    return psf

# def deblur_if_needed(img_bgr, threshold=100.0):
#     """
#     If blur metric is below threshold, apply light deconvolution.
#     Returns image in uint8 BGR
#     """
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     metric = blur_metric_gray(gray)
#     print(f"[deblur] blur metric: {metric:.2f}")
#     if metric < threshold:
#         # convert to float [0,1]
#         imgf = img_as_float(img_rgb)
#         psf = estimate_motion_psf(length=15, angle=0)
#         # apply RL on each channel
#         out = np.zeros_like(imgf)
#         for c in range(3):
#             out[..., c] = richardson_lucy(imgf[..., c], psf, num_iter=10)
#         out = (out * 255.0).clip(0,255).astype("uint8")
#         out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
#         print("[deblur] applied richardson-lucy deconvolution")
#     else:
#         out_bgr = img_bgr
#     # denoise lightly
#     out_bgr = cv2.fastNlMeansDenoisingColored(out_bgr, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
#     out_bgr = normalize_uint8(out_bgr)
#     return out_bgr

def deblur_if_needed(img_bgr, threshold=100.0):
    # Instead of complex deconvolution, we use light denoising
    # This keeps skin looking smooth and professional
    denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21)
    return denoised