import os
import cv2
import numpy as np
from mediapipe.python.solutions import pose as mp_pose


def _detect_shoulder_line(
    img_rgb: np.ndarray,
    height: int,
    shoulder_scale: float,
    fallback_ratio: float,
) -> int:
    with mp_pose.Pose(static_image_mode=True) as pose_detector:
        pose_results = pose_detector.process(img_rgb)
        if not pose_results.pose_landmarks:
            return int(height * fallback_ratio)

        l_shldr = pose_results.pose_landmarks.landmark[11].y * height
        r_shldr = pose_results.pose_landmarks.landmark[12].y * height
        line = int(max(l_shldr, r_shldr) * shoulder_scale)
        return min(max(line, 1), height)


def crop_to_shoulder(
    person_path: str,
    output_path: str,
    shoulder_scale: float = 1.55,
    fallback_ratio: float = 0.5,
) -> None:
    img = cv2.imread(person_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {person_path}")

    h, w = img.shape[:2]

    # For pose detection, use RGB (ignoring alpha if present)
    bgr = img[:, :, :3]
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    shoulder_line = _detect_shoulder_line(img_rgb, h, shoulder_scale, fallback_ratio)

    cropped = img[0:shoulder_line, 0:w]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, cropped)


if __name__ == "__main__":
    input_dir = "output/transparent"         
    output_dir = "output/crop_till_shoulder"
    total_images = 54

    for i in range(50, total_images + 1):
        input_path = os.path.join(input_dir, f"result_{i}.png")
        if not os.path.exists(input_path):
            print(f"[{i}/{total_images}] Skip: {input_path} not found")
            continue

        output_path = os.path.join(output_dir, f"crop_{i}.png")
        print(f"[{i}/{total_images}] Processing: {input_path}...", end=" ", flush=True)

        crop_to_shoulder(
            person_path=input_path,
            output_path=output_path,    
            shoulder_scale=1.15,
            fallback_ratio=0.5,
        )

        print(f"Done! Saved to {output_path}")
