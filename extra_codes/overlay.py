import os
import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def find_shoulders(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]

    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark

    left = (int(lm[11].x * w), int(lm[11].y * h))
    right = (int(lm[12].x * w), int(lm[12].y * h))

    return {
        "left_shoulder": left,
        "right_shoulder": right,
        "midpoint": ((left[0]+right[0])//2, (left[1]+right[1])//2),
        "width": np.linalg.norm(np.array(left) - np.array(right))
    }

def overlay_rgba(background, overlay, x, y):
    bg = background.copy()

    h, w = overlay.shape[:2]

    # Clip overlay region if it goes out of bounds
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bg.shape[1]), min(y + h, bg.shape[0])

    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return bg

    alpha = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
    for c in range(3):
        bg[y1:y2, x1:x2, c] = (
            alpha * overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c]
            + (1 - alpha) * bg[y1:y2, x1:x2, c]
        )

    return bg

# ------------------ Load template once ------------------

template_path = "output/template/template_0_1696_2528.png"
template_img = cv2.imread(template_path)
template_info = find_shoulders(template_path)

if template_img is None or template_info is None:
    raise ValueError("Template image or shoulders not detected")

if template_img.shape[2] == 4:
    template_img = template_img[:, :, :3]

ltx, lty = template_info["left_shoulder"]
rtx, rty = template_info["right_shoulder"]
template_width = template_info["width"]

# Output folder
os.makedirs("output/on_template", exist_ok=True)

# ------------------ Loop over user images ------------------

template_without_human = cv2.imread("output/template/template_not_human_1696_2528.png")
# template_without_human_scoreboard = cv2.imread("output/template/template_not_human_scoreboard_1696_2528.png")

for i in range(50, 55):
    user_path = f"output/transparent/result_{i}.png"
    user_img = cv2.imread(user_path, cv2.IMREAD_UNCHANGED)
    user_info = find_shoulders(user_path)

    if user_img is None or user_info is None:
        print(f"[SKIP] result_{i}.png → shoulders not detected")
        continue

    lux, luy = user_info["left_shoulder"]
    rux, ruy = user_info["right_shoulder"]
    user_width = user_info["width"]

    scale = template_width / user_width

    # print(f"user_{i}: {scale}")

    uh, uw = user_img.shape[:2]
    scaled_user = cv2.resize(
        user_img,
        (int(uw * scale), int(uh * scale)),
        interpolation=cv2.INTER_LINEAR
    )

    scaled_h, scaled_w = scaled_user.shape[:2]
    print(f"user_{i}: {scaled_h}")

    lux_s, luy_s = int(lux * scale), int(luy * scale)

    dx = ltx - lux_s
    dy = lty - luy_s

    final_img = overlay_rgba(template_without_human, scaled_user, dx, dy)

    out_path = f"output/on_template_without_human/final_{i}.png"
    cv2.imwrite(out_path, final_img)

    print(f"[OK] Saved {out_path}")

cv2.destroyAllWindows()