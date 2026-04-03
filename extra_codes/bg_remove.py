import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

def remove_background(person_img, session, output_path: str) -> None:
    if person_img is None:
        raise FileNotFoundError("Input image is empty or could not be read.")
    h, w, _ = person_img.shape

    # MODNet preprocess (expects 512x512 RGB, normalized to [-1, 1])
    img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(img_rgb, (512, 512))
    input_img = (input_img.astype(np.float32) - 127.5) / 127.5
    input_img = np.transpose(input_img, (2, 0, 1))[None, ...]

    input_name = session.get_inputs()[0].name
    matte = session.run(None, {input_name: input_img})[0]
    matte = np.squeeze(matte)
    matte = cv2.resize(matte, (w, h))
    matte = np.clip(matte, 0.0, 1.0)

    alpha = (matte * 255).astype(np.uint8)
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]

    bgra = np.concatenate([person_img, alpha], axis=2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, bgra)

if __name__ == "__main__":
    model_path = "model/modnet_photographic_portrait_matting.onnx"
    
    # Initialize the session once to save time
    print(f"--- Loading model: {model_path} ---")
    session = ort.InferenceSession(model_path)
    
    total_images = 54
    for i in range(50, total_images + 1):
        base_path = f"assets/input_{i}"
        person_path = None
    
        for ext in [".png", ".jpg", ".jpeg"]:
            potential_path = Path(f"{base_path}{ext}")
            if potential_path.exists():
                person_path = str(potential_path)
                break 

        if person_path:
            print(f"[{i}/{total_images}] Processing: {person_path}...", end=" ", flush=True)
            
            img = cv2.imread(person_path)
            output_file = f"output/transparent/result_{i}.png"
            
            remove_background(img, session, output_file)
            
            print(f"Done! Saved to {output_file}")
        else:
            print(f"[{i}/{total_images}] Skip: No file found for input_{i}")

    print("--- All tasks complete! ---")
