# import cv2
# import numpy as np
# import os
# import onnxruntime as ort
# import mediapipe as mp

# # Direct imports for MediaPipe submodules
# from mediapipe.python.solutions import pose as mp_pose

# def process_to_500x500_square(person_path, model_path, output_path):
#     # 1. Load Source Image
#     person_img = cv2.imread(person_path)
#     if person_img is None:
#         print(f"Error: Could not load {person_path}")
#         return
        
#     session = ort.InferenceSession(model_path)
#     h, w, _ = person_img.shape

#     # 2. MODNet Pre-processing (Alpha Matte Extraction)
#     img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
#     input_img = cv2.resize(img_rgb, (512, 512))
#     input_img = (input_img.astype(np.float32) - 127.5) / 127.5
#     input_img = np.transpose(input_img, (2, 0, 1))
#     input_img = np.expand_dims(input_img, axis=0)

#     # 3. Get Alpha Matte from MODNet
#     input_name = session.get_inputs()[0].name
#     matte = session.run(None, {input_name: input_img})[0]
#     matte = np.squeeze(matte)
#     matte = cv2.resize(matte, (w, h))
#     matte = np.expand_dims(matte, axis=-1)

#     # 4. Detect Shoulder Line for Head-to-Shoulder Crop
#     with mp_pose.Pose(static_image_mode=True) as pose_detector:
#         pose_results = pose_detector.process(img_rgb)
#         if not pose_results.pose_landmarks:
#             shoulder_line = int(h * 0.5) # Fallback to 50% height
#         else:
#             # Landmark 11 & 12 are left/right shoulders
#             l_shldr = pose_results.pose_landmarks.landmark[11].y * h
#             r_shldr = pose_results.pose_landmarks.landmark[12].y * h
#             shoulder_line = int(max(l_shldr, r_shldr) * 1.15) # 15% margin below shoulders
#             shoulder_line = min(shoulder_line, h)

#     # 5. Extract Cutout and Apply Matte
#     # This removes the background by multiplying pixels by the alpha values (0 to 1)
#     foreground = person_img.astype(float) * matte
#     cutout = foreground[0:shoulder_line, 0:w]
#     alpha_part = matte[0:shoulder_line, 0:w]

#     # 6. Resize Cutout to Width 600 (Maintain Aspect Ratio)
#     scale = 600 / w
#     new_h = int(cutout.shape[0] * scale)
    
#     resized_cutout = cv2.resize(cutout, (600, new_h))
#     resized_alpha = cv2.resize(alpha_part, (600, new_h))
#     resized_alpha = np.expand_dims(resized_alpha, axis=-1)

#     # 7. Create 600x600 Black Background and Paste
#     final_canvas = np.zeros((600, 600, 3), dtype=np.float32)
    
#     # Calculate paste coordinates to match the base
#     # If the person is taller than 600px, we crop the top.
#     # If shorter, they will sit at the bottom of the black canvas.
#     start_y_canvas = max(0, 600 - new_h)
#     start_y_cutout = max(0, new_h - 600)
#     h_to_paste = min(new_h, 600)

#     # Slice the source and target areas
#     target_area = final_canvas[start_y_canvas:600, 0:600]
#     source_pixels = resized_cutout[start_y_cutout:start_y_cutout + h_to_paste, :]
#     source_alpha = resized_alpha[start_y_cutout:start_y_cutout + h_to_paste, :]
    
#     # Composite: Since canvas is black (0), it is effectively just the source_pixels
#     # But this blending logic handles the soft edges correctly
#     final_canvas[start_y_canvas:600, 0:600] = source_pixels + target_area * (1 - source_alpha)

#     # 8. Save Final Output
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     cv2.imwrite(output_path, final_canvas.astype(np.uint8))
#     print(f"Success! Processed portrait saved to: {output_path}")

# # if __name__ == "__main__":
# #     process_to_500x500_square(
# #         person_path="assets/input_8.png",
# #         model_path="model/modnet_photographic_portrait_matting.onnx",
# #         output_path="output/500_500/MODNet_8.jpg"
# #     )

# import cv2
# import numpy as np
# import os
# import onnxruntime as ort

# def process_to_700x700_square(person_path, model_path, output_path):
#     # 1. Load Source Image
#     person_img = cv2.imread(person_path)
#     if person_img is None:
#         print(f"Error: Could not load {person_path}")
#         return
        
#     session = ort.InferenceSession(model_path)
#     h, w, _ = person_img.shape

#     # 2. MODNet Pre-processing
#     img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
#     input_img = cv2.resize(img_rgb, (512, 512))
#     input_img = (input_img.astype(np.float32) - 127.5) / 127.5
#     input_img = np.transpose(input_img, (2, 0, 1))
#     input_img = np.expand_dims(input_img, axis=0)

#     # 3. Get Alpha Matte from MODNet
#     input_name = session.get_inputs()[0].name
#     matte = session.run(None, {input_name: input_img})[0]
#     matte = np.squeeze(matte)
#     matte = cv2.resize(matte, (w, h))
    
#     # 4. Determine Crop Line via Alpha Matte (Replacement for MediaPipe)
#     # Find all rows where the person exists (alpha > 0.1)
#     alpha_sum = np.sum(matte, axis=1)
#     nonzero_rows = np.where(alpha_sum > 0.1)[0]
    
#     if len(nonzero_rows) > 0:
#         # Define the bottom crop at the lowest detected part of the person
#         # or use a fixed ratio of the detected height for a "portrait" look
#         top_bound = nonzero_rows[0]
#         bottom_bound = nonzero_rows[-1]
#         detected_height = bottom_bound - top_bound
        
#         # Heuristic: Crop at 70% of the detected height to simulate a shoulder crop
#         shoulder_line = int(top_bound + (detected_height * 0.7))
#     else:
#         shoulder_line = int(h * 0.5) # Fallback

#     shoulder_line = min(shoulder_line, h)
#     matte_expanded = np.expand_dims(matte, axis=-1)

#     # 5. Extract Cutout
#     foreground = person_img.astype(float) * matte_expanded
#     cutout = foreground[0:shoulder_line, 0:w]
#     alpha_part = matte_expanded[0:shoulder_line, 0:w]

#     # 6. Resize Cutout to Width 700
#     scale = 700 / w
#     new_h = int(cutout.shape[0] * scale)
    
#     resized_cutout = cv2.resize(cutout, (700, new_h))
#     resized_alpha = cv2.resize(alpha_part, (700, new_h))
#     if len(resized_alpha.shape) == 2:
#         resized_alpha = np.expand_dims(resized_alpha, axis=-1)

#     # 7. Create 700x700 Black Background and Paste
#     final_canvas = np.zeros((700, 700, 3), dtype=np.float32)
    
#     start_y_canvas = max(0, 700 - new_h)
#     start_y_cutout = max(0, new_h - 700)
#     h_to_paste = min(new_h, 700)

#     target_area = final_canvas[start_y_canvas:700, 0:700]
#     source_pixels = resized_cutout[start_y_cutout:start_y_cutout + h_to_paste, :]
#     source_alpha = resized_alpha[start_y_cutout:start_y_cutout + h_to_paste, :]
    
#     final_canvas[start_y_canvas:700, 0:700] = source_pixels + target_area * (1 - source_alpha)

#     # 8. Save Final Output
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     cv2.imwrite(output_path, final_canvas.astype(np.uint8))

# if __name__ == "__main__":
#     output_dir = "output/700_700"
#     os.makedirs(output_dir, exist_ok=True)

#     for i in range(1, 37):
#         path_png = os.path.join("assets", f"input_{i}.png")
#         path_jpg = os.path.join("assets", f"input_{i}.jpg")

#         if os.path.exists(path_png):
#             person_path = path_png
#         elif os.path.exists(path_jpg):
#             person_path = path_jpg
#         else:
#             print(f"Skipping: input_{i} (No .png or .jpg found)")
#             continue

#         output_path = os.path.join(output_dir, f"MODNet_{i}.jpg")
        
#         print(f"Processing: {os.path.basename(person_path)}...")
#         process_to_700x700_square(
#             person_path=person_path,
#             model_path="model/modnet_photographic_portrait_matting.onnx",
#             output_path=output_path
#         )

import cv2
import numpy as np
import os
import onnxruntime as ort

def process_to_black_bg(person_path, model_path, output_path):
    # 1. Load Source Image
    person_img = cv2.imread(person_path)
    if person_img is None:
        print(f"Error: Could not load {person_path}")
        return
        
    session = ort.InferenceSession(model_path)
    h, w, _ = person_img.shape

    # 2. MODNet Pre-processing
    img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(img_rgb, (512, 512))
    input_img = (input_img.astype(np.float32) - 127.5) / 127.5
    input_img = np.transpose(input_img, (2, 0, 1))
    input_img = np.expand_dims(input_img, axis=0)

    # 3. Get Alpha Matte from MODNet
    input_name = session.get_inputs()[0].name
    matte = session.run(None, {input_name: input_img})[0]
    matte = np.squeeze(matte)
    matte = cv2.resize(matte, (w, h))
    matte = np.expand_dims(matte, axis=-1) # Shape (h, w, 1)

    # 4. Apply Matte to Background
    # Multiplying by matte keeps the person and turns everything else to 0 (Black)
    final_img = (person_img.astype(np.float32) * matte).astype(np.uint8)

    # 5. Save Final Output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final_img)
    print(f"Success: {os.path.basename(output_path)}")

if __name__ == "__main__":
    output_dir = "output/black_bg"
    model_file = "model/modnet_photographic_portrait_matting.onnx"
    
    for i in range(1, 37):
        # Check for png or jpg
        for ext in [".png", ".jpg"]:
            person_path = os.path.join("assets", f"input_{i}{ext}")
            if os.path.exists(person_path):
                output_path = os.path.join(output_dir, f"MODNet_{i}.jpg")
                process_to_black_bg(person_path, model_file, output_path)
                break