import os
import cv2
import mediapipe as mp
import numpy as np

def apply_face_glow(image_path, output_path, brightness_factor=1.8, threshold=120):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    img = cv2.imread(image_path)
    if img is None: return
    h, w, _ = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detection.process(img_rgb)
    
    if not results.detections:
        print(f"No face detected in {os.path.basename(image_path)}.")
        cv2.imwrite(output_path, img) 
        return

    # Initialize mask and flags
    mask = np.zeros((h, w), dtype=np.uint8)
    should_brighten = False
    brightness_values = [] # Store brightness of all faces detected

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x, y, fw, fh = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
        
        # Ensure coordinates are within image boundaries and valid
        x, y = max(0, x), max(0, y)
        fw, fh = max(1, fw), max(1, fh) # Ensure width/height are at least 1
        
        face_roi = img[y:y+fh, x:x+fw]

        if face_roi.size == 0: continue # Skip invalid faces

        # Calculate average brightness using the L channel
        avg_brightness = np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)[:, :, 0])
        brightness_values.append(avg_brightness)

        if avg_brightness < threshold:
            should_brighten = True

        # --- Create Graduated Mask for smoother blending ---
        center = (x + fw // 2, y + fh // 2)
        
        # 1. Outer, fainter glow layer (wider than face)
        axes_outer = (int(fw * 0.8), int(fh * 1.0))
        cv2.ellipse(mask, center, axes_outer, 0, 0, 360, 80, -1) # Low intensity

        # 2. Middle glow layer
        axes_mid = (int(fw * 0.6), int(fh * 0.8))
        cv2.ellipse(mask, center, axes_mid, 0, 0, 360, 160, -1) # Medium intensity

        # 3. Inner core layer (brightest part of face)
        axes_inner = (int(fw * 0.4), int(fh * 0.6))
        cv2.ellipse(mask, center, axes_inner, 0, 0, 360, 255, -1) # Full intensity

    if not should_brighten or not brightness_values:
        avg_b = np.mean(brightness_values) if brightness_values else 0
        print(f"Skipping glow for {os.path.basename(image_path)} (Avg Brightness: {avg_b:.2f})")
        cv2.imwrite(output_path, img)
        return

    # Apply Heavier Gaussian Blur to melt the graduated ellipses together
    # Using a very large kernel (151, 151) for extreme softness
    mask_blurred = cv2.GaussianBlur(mask, (151, 151), 0) / 255.0
    mask_blurred = np.stack([mask_blurred] * 3, axis=-1)

    # Apply Gamma Correction
    gamma = 1.0 / brightness_factor
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    bright_img = cv2.LUT(img, lut)

    # Blend using the new super-soft mask
    final_img = (img * (1 - mask_blurred) + bright_img * mask_blurred).astype(np.uint8)
    
    avg_b = np.mean(brightness_values)
    cv2.imwrite(output_path, final_img)
    print(f"Applied smooth glow to {os.path.basename(image_path)} (Avg Brightness was: {avg_b:.2f})")

# --- Execution loop ---
# Make sure inputs and outputs exist
os.makedirs("output/on_template_without_human", exist_ok=True)
os.makedirs("output/Face_improve", exist_ok=True)

print("Starting batch processing...")
for i in range(1, 55):
    in_p = f"output/on_template_without_human/final_{i}.png"
    out_p = f"output/Face_improve/final_{i}.png"
    
    if os.path.exists(in_p):
        # You might want to lower threshold slightly if it's still too aggressive, e.g., 110
        apply_face_glow(in_p, out_p, brightness_factor=1.8, threshold=120)
    else:
        print(f"Input file not found: {in_p}")

print("Batch processing complete.")