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
        self.session = ort.InferenceSession(modnet_path)
        self.template_bg = cv2.imread(template_bg_path)
        if self.template_bg is None:
            raise FileNotFoundError(f"Template background not found: {template_bg_path}")
        
        print(f"[Scorecard] Calculating template shoulder reference from {template_with_person_path}...")
        self.template_info = self._find_shoulders(template_with_person_path)
        if self.template_info is None:
            raise ValueError("Could not detect shoulders in the template image. Please check the template reference.")
        
        self.ltx, self.lty = self.template_info["left_shoulder"]
        self.rtx, self.rty = self.template_info["right_shoulder"]
        self.template_width = self.template_info["width"]

    def _find_shoulders(self, image_input):
        """
        Detects left and right shoulders using MediaPipe Pose.
        image_input can be a file path or a BGR numpy array.
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        else:
            img = image_input

        if img is None:
            return None

        h, w = img.shape[:2]
        with mp_pose.Pose(static_image_mode=True) as pose:
            # MediaPipe expects RGB
            res = pose.process(cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB))

        if not res.pose_landmarks:
            return None

        lm = res.pose_landmarks.landmark
        left = (int(lm[11].x * w), int(lm[11].y * h))
        right = (int(lm[12].x * w), int(lm[12].y * h))

        return {
            "left_shoulder": left,
            "right_shoulder": right,
            "width": np.linalg.norm(np.array(left) - np.array(right))
        }

    def remove_background(self, img_bgr):
        """
        Removes background using MODNet and returns a BGRA image.
        """
        h, w = img_bgr.shape[:2]
        # MODNet Pre-processing (512x512 RGB normalized to [-1, 1])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(img_rgb, (512, 512))
        input_img = (input_img.astype(np.float32) - 127.5) / 127.5
        input_img = np.transpose(input_img, (2, 0, 1))[None, ...]

        input_name = self.session.get_inputs()[0].name
        matte = self.session.run(None, {input_name: input_img})[0]
        matte = np.squeeze(matte)
        matte = cv2.resize(matte, (w, h))
        matte = np.clip(matte, 0.0, 1.0)

        alpha = (matte * 255).astype(np.uint8)
        if alpha.ndim == 2:
            alpha = alpha[:, :, None]

        # Combine BGR and Alpha
        bgra = np.concatenate([img_bgr, alpha], axis=2)
        return bgra

    def overlay_on_template(self, user_bgra):
        """
        Rescales user image based on shoulders and overlays onto the template.
        """
        user_info = self._find_shoulders(user_bgra)
        if user_info is None:
            print("[Scorecard] Warning: Shoulders not detected in generated image. Skipping overlay.")
            return None

        lux, luy = user_info["left_shoulder"]
        user_width = user_info["width"]

        # Calculate scale to match template shoulder width
        scale = self.template_width / user_width
        
        uh, uw = user_bgra.shape[:2]
        scaled_user = cv2.resize(
            user_bgra,
            (int(uw * scale), int(uh * scale)),
            interpolation=cv2.INTER_LINEAR
        )

        lux_s, luy_s = int(lux * scale), int(luy * scale)
        dx = self.ltx - lux_s
        dy = self.lty - luy_s

        # Overlay logic
        bg = self.template_bg.copy()
        h, w = scaled_user.shape[:2]

        # Clip overlay region if it goes out of bounds
        x1, y1 = max(dx, 0), max(dy, 0)
        x2, y2 = min(dx + w, bg.shape[1]), min(dy + h, bg.shape[0])

        overlay_x1 = max(0, -dx)
        overlay_y1 = max(0, -dy)
        overlay_x2 = overlay_x1 + (x2 - x1)
        overlay_y2 = overlay_y1 + (y2 - y1)

        if x1 >= x2 or y1 >= y2:
            return bg

        alpha = scaled_user[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
        for c in range(3):
            bg[y1:y2, x1:x2, c] = (
                alpha * scaled_user[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c]
                + (1 - alpha) * bg[y1:y2, x1:x2, c]
            )

        return bg
