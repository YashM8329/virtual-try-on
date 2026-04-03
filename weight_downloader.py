"""
weight_downloader.py
Downloads all required pre-trained model weights:
- SAM ViT-H checkpoint (~2.4 GB)
- MediaPipe Pose Landmarker Heavy model
- MediaPipe Selfie Multiclass segmentation model
"""

import os
import urllib.request


SAM_CHECKPOINT = "weights/sam_vit_h_4b8939.pth"
POSE_LANDMARKER_MODEL = "weights/pose_landmarker_heavy.task"
MP_MODEL_PATH = "weights/selfie_multiclass_256x256.tflite"


def download_weights():
    os.makedirs("weights", exist_ok=True)

    # SAM ViT-H checkpoint (~2.4 GB)
    if not os.path.exists(SAM_CHECKPOINT):
        print("Downloading SAM ViT-H weights (~2.4 GB, please wait)...")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            SAM_CHECKPOINT,
        )
        print("SAM weights downloaded.")
    else:
        print("✅ SAM weights already present.")

    # MediaPipe Pose Landmarker Heavy
    if not os.path.exists(POSE_LANDMARKER_MODEL):
        print("Downloading MediaPipe Pose Landmarker model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
            POSE_LANDMARKER_MODEL,
        )
        print("Pose Landmarker model downloaded.")
    else:
        print("✅ Pose Landmarker model already present.")

    # MediaPipe Selfie Multiclass segmentation model
    if not os.path.exists(MP_MODEL_PATH):
        print("Downloading MediaPipe Selfie Multiclass model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
            MP_MODEL_PATH,
        )
        print("MediaPipe model downloaded.")
    else:
        print("✅ MediaPipe Selfie Multiclass model already present.")

    print("\n✅ All weights ready.")


if __name__ == "__main__":
    download_weights()
