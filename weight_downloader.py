"""
weight_downloader.py
Downloads only the required pre-trained model weights for the optimized pipeline.
SAM ViT-H (~2.4 GB) is excluded as it is no longer used in the high-speed flow.
"""

import os
import urllib.request

POSE_LANDMARKER_MODEL = "weights/pose_landmarker_heavy.task"
MP_MODEL_PATH = "weights/selfie_multiclass_256x256.tflite"
GFPGAN_MODEL_PATH = "weights/GFPGANv1.4.pth"

def download_weights():
    os.makedirs("weights", exist_ok=True)

    # MediaPipe Pose Landmarker Heavy
    if not os.path.exists(POSE_LANDMARKER_MODEL):
        print("Downloading MediaPipe Pose Landmarker model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
            POSE_LANDMARKER_MODEL,
        )
        print("Pose Landmarker model downloaded.")

    # MediaPipe Selfie Multiclass segmentation model
    if not os.path.exists(MP_MODEL_PATH):
        print("Downloading MediaPipe Selfie Multiclass model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
            MP_MODEL_PATH,
        )
        print("MediaPipe model downloaded.")

    # GFPGAN v1.4 weights
    if not os.path.exists(GFPGAN_MODEL_PATH):
        print("Downloading GFPGAN v1.4 weights...")
        urllib.request.urlretrieve(
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            GFPGAN_MODEL_PATH,
        )
        print("GFPGAN weights downloaded.")

if __name__ == "__main__":
    download_weights()
