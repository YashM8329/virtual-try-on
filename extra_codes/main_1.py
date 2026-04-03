import cv2
import numpy as np
import os
import onnxruntime as ort

def MOEDNet(person_path, model_path, output_path):

    person_img = cv2.imread(person_path)
    if person_img is None:
        print(f"Error: Could not load {person_path}")
        return
    
    session = ort.InferenceSession(model_path)
    h, w, _ = person_img.shape

    