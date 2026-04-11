import os
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
from PIL import Image

mp_pose = mp.solutions.pose

class ScorecardProcessor:
    def __init__(self, modnet_path, template_with_person_path, template_bg_path):
        """
        Initializes the ScorecardProcessor with MODNet and template info.
        """
        print(f"[Scorecard] Initializing MODNet from {modnet_path}...")
        # Use CUDA if available for ~10x speedup on background removal
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(modnet_path, providers=providers)
        
        self.template_bg = cv2.imread(template_bg_path)
        if self.template_bg is None:
            raise FileNotFoundError(f"Template background not found: {template_bg_path}")
        
        print(f"[Scorecard] Calculating template shoulder reference...")
        self.template_info = self._find_shoulders(template_with_person_path)
        if self.template_info is None:
            raise ValueError("Could not detect shoulders in template.")
        
        self.ltx, self.lty = self.template_info["left_shoulder"]
        self.rtx, self.rty = self.template_info["right_shoulder"]
        self.template_width = self.template_info["width"]

    def _find_shoulders(self, image_input):
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        else:
            img = image_input
        if img is None: return None

        h, w = img.shape[:2]
        with mp_pose.Pose(static_image_mode=True) as pose:
            res = pose.process(cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks: return None

        lm = res.pose_landmarks.landmark
        left = (int(lm[11].x * w), int(lm[11].y * h))
        right = (int(lm[12].x * w), int(lm[12].y * h))
        return {
            "left_shoulder": left,
            "right_shoulder": right,
            "width": np.linalg.norm(np.array(left) - np.array(right))
        }

    def remove_background(self, img_bgr):
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(img_rgb, (512, 512))
        input_img = (input_img.astype(np.float32) - 127.5) / 127.5
        input_img = np.transpose(input_img, (2, 0, 1))[None, ...]

        matte = self.session.run(None, {self.session.get_inputs()[0].name: input_img})[0]
        matte = cv2.resize(np.squeeze(matte), (w, h))
        alpha = (np.clip(matte, 0, 1) * 255).astype(np.uint8)
        return np.concatenate([img_bgr, alpha[:, :, None]], axis=2)

    def overlay_on_template(self, user_bgra, user_landmarks=None):
        """
        user_landmarks: If provided, skips MediaPipe re-detection (huge speedup for 41 images)
        """
        if user_landmarks:
            h, w = user_bgra.shape[:2]
            left = user_landmarks[11]
            right = user_landmarks[12]
            user_info = {
                "left_shoulder": left,
                "right_shoulder": right,
                "width": np.linalg.norm(np.array(left) - np.array(right))
            }
        else:
            user_info = self._find_shoulders(user_bgra)
            
        if user_info is None: return None

        lux, luy = user_info["left_shoulder"]
        scale = self.template_width / user_info["width"]
        
        uh, uw = user_bgra.shape[:2]
        scaled_user = cv2.resize(user_bgra, (int(uw * scale), int(uh * scale)), interpolation=cv2.INTER_LINEAR)

        dx, dy = self.ltx - int(lux * scale), self.lty - int(luy * scale)
        bg = self.template_bg.copy()
        sh, sw = scaled_user.shape[:2]

        x1, y1 = max(dx, 0), max(dy, 0)
        x2, y2 = min(dx + sw, bg.shape[1]), min(dy + sh, bg.shape[0])
        ox1, oy1 = max(0, -dx), max(0, -dy)
        ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

        if x1 < x2 and y1 < y2:
            alpha = scaled_user[oy1:oy2, ox1:ox2, 3] / 255.0
            for c in range(3):
                bg[y1:y2, x1:x2, c] = alpha * scaled_user[oy1:oy2, ox1:ox2, c] + (1 - alpha) * bg[y1:y2, x1:x2, c]
        return bg
