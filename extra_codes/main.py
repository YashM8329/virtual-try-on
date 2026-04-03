import cv2
import numpy as np
import os
import onnxruntime as ort
import mediapipe as mp

# Keeping MediaPipe Pose for the shoulder-line detection
from mediapipe.python.solutions import pose as mp_pose

def process_portrait(person_path, bg_path, model_path, output_path):
    person_img = cv2.imread(person_path)
    black_bg = cv2.imread(bg_path)
    
    if person_img is None or black_bg is None:
        print(f"Error: Could not load images. Check paths.")
        return
        
    session = ort.InferenceSession(model_path)

    # Standardize background to 1280x720
    black_bg = cv2.resize(black_bg, (720, 1280))
    h, w, _ = person_img.shape

    # MODNet Pre-processing
    # MODNet expects a 512x512 normalized RGB input
    img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(img_rgb, (512, 512))
    input_img = (input_img.astype(np.float32) - 127.5) / 127.5
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = np.expand_dims(input_img, axis=0)

    # MODNet Inference (Get Alpha Matte)
    input_name = session.get_inputs()[0].name
    matte = session.run(None, {input_name: input_img})[0]
    
    # Resize matte back to original image size
    matte = np.squeeze(matte)
    matte = cv2.resize(matte, (w, h))
    matte = np.expand_dims(matte, axis=-1) # Shape (h, w, 1)

    # Detect Shoulder Line (Using your preferred Pose logic)
    with mp_pose.Pose(static_image_mode=True) as pose_detector:
        pose_results = pose_detector.process(img_rgb)
        if not pose_results.pose_landmarks:
            print("No human detected. Defaulting to middle crop.")
            shoulder_line = int(h * 0.5)
        else:
            # Landmark 11 & 12 are shoulders
            l_shldr = pose_results.pose_landmarks.landmark[11].y * h
            r_shldr = pose_results.pose_landmarks.landmark[12].y * h
            shoulder_line = int(max(l_shldr, r_shldr) * 1.15)
            shoulder_line = min(shoulder_line, h)

    # Extraction with Alpha Blending
    # Multiplying by matte creates soft edges
    foreground = person_img.astype(float) * matte 
    
    portrait_cutout = foreground[0:shoulder_line, 0:w]
    portrait_alpha = matte[0:shoulder_line, 0:w]

    # Resize to fit 720px width background
    aspect_ratio = portrait_cutout.shape[0] / portrait_cutout.shape[1]
    new_w = 720
    new_h = int(new_w * aspect_ratio)
    
    resized_cutout = cv2.resize(portrait_cutout, (new_w, new_h))
    resized_alpha = cv2.resize(portrait_alpha, (new_w, new_h))
    resized_alpha = np.expand_dims(resized_alpha, axis=-1)

    final_output = black_bg.copy().astype(float)
    paste_h = min(new_h, 1280)
    
    # Formula: Foreground + Background * (1 - Alpha)
    # This ensures the person blends into the black instead of just sitting on top
    target_bg_area = final_output[0:paste_h, 0:720]
    composite = resized_cutout[0:paste_h, :] + target_bg_area * (1 - resized_alpha[0:paste_h, :])
    
    final_output[0:paste_h, 0:720] = composite

    # Save as uint8
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final_output.astype(np.uint8))
    print(f"MODNet process complete. Saved to: {output_path}")

if __name__ == "__main__":
    process_portrait(
        person_path="assets/input_5.png", 
        bg_path="public/background.jpg",
        model_path="assets/modnet_photographic_portrait_matting.onnx",
        output_path="output/result_5.jpg"
    )