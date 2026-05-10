import cv2
import os
import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────

def read_yolo_label(lbl_path):
    with open(lbl_path, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    cls, xc, yc, wn, hn = lines[0].split()[:5]
    return int(cls), float(xc), float(yc), float(wn), float(hn)

def yolo_to_xywh(xc, yc, wn, hn, W, H):
    w = wn * W
    h = hn * H
    x = xc * W - w / 2
    y = yc * H - h / 2
    return (x, y, w, h)

def xywh_to_yolo(x, y, w, h, W, H):
    xc = (x + w / 2) / W
    yc = (y + h / 2) / H
    wn = w / W
    hn = h / H
    return (xc, yc, wn, hn)

def apply_affine_to_bbox_xywh(b, M):
    x, y, w, h = b
    pts = np.array([
        [x,     y],
        [x+w,   y],
        [x,   y+h],
        [x+w, y+h]
    ], dtype=np.float32).reshape(-1, 1, 2)
    pts_t = cv2.transform(pts, M).reshape(-1, 2)
    x_min = float(np.min(pts_t[:, 0]))
    y_min = float(np.min(pts_t[:, 1]))
    x_max = float(np.max(pts_t[:, 0]))
    y_max = float(np.max(pts_t[:, 1]))
    return (x_min, y_min, x_max - x_min, y_max - y_min)

def union_bbox(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    x_min = min(x1, x2)
    y_min = min(y1, y2)
    x_max = max(x1 + w1, x2 + w2)
    y_max = max(y1 + h1, y2 + h2)
    return (x_min, y_min, x_max - x_min, y_max - y_min)

# ── Config ────────────────────────────────────────────────────────────────────

BALL_CLASS_ID = 0
IMG_DIR = r"D:\dataset_new_split\images"
LBL_DIR = r"D:\dataset_new_split\labels"

img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png"))])


for i in range(len(img_files) - 1):
    img_name1 = img_files[i]
    img_name2 = img_files[i + 1]

    stem1 = os.path.splitext(img_name1)[0]
    stem2 = os.path.splitext(img_name2)[0]

    img1 = cv2.imread(os.path.join(IMG_DIR, img_name1))
    img2 = cv2.imread(os.path.join(IMG_DIR, img_name2))

    lab1 = read_yolo_label(os.path.join(LBL_DIR, stem1 + ".txt"))
    lab2 = read_yolo_label(os.path.join(LBL_DIR, stem2 + ".txt"))

    gray1_u8 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2_u8 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    gray1 = gray1_u8.astype(np.float32)
    gray2 = gray2_u8.astype(np.float32)

    A = gray1 - gray2 #   (-255...255)
    B = (A + 255.0) / 2.0 #  (0...255)
    gray_diff = np.clip(B, 0, 255).astype(np.uint8) # force any values into the range (0-255)
    H, W = gray_diff.shape

    cls1, xc1, yc1, wn1, hn1 = lab1
    cls2, xc2, yc2, wn2, hn2 = lab2

    b1 = yolo_to_xywh(xc1, yc1, wn1, hn1, W, H)
    b2 = yolo_to_xywh(xc2, yc2, wn2, hn2, W, H)


    union = union_bbox(b1, b2)
    x, y, w, h = union
    pad = int(max(2, min(6, 0.10 * max(w, h))))  # 10% of size
    x -= pad           # tilt top left corner, add padding to both sides
    y -= pad
    w += 2 * pad
    h += 2 * pad
    x = max(0, x)    # ensure bbox is in the frame
    y = max(0, y)
    w = min(W - x, w)
    h = min(H - y, h)

    xc, yc, wn, hn = xywh_to_yolo(x, y, w, h, W, H)
    # Visualization
    bbox_vis = cv2.cvtColor(gray_diff, cv2.COLOR_GRAY2BGR) # convert back to 3-channels to draw a bbox
    cv2.rectangle(bbox_vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)  # draw a bbox

    img_name = f"{stem1}.jpg"
    lbl_name = f"{stem1}.txt"

    cv2.imwrite(os.path.join("dataset_new_diff/images/train", img_name), gray_diff)
    cv2.imwrite(os.path.join("dataset_new_diff/bounding_box/train", img_name), bbox_vis)

    with open(os.path.join("dataset_new_diff/labels/train", lbl_name), "w") as f:
        f.write(f"{BALL_CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    print(f"{i+1}/{len(img_files)-1}")

