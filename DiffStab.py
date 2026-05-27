import cv2
import os
import numpy as np

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


def xywh_to_yolo(x, y, w, h, W, H):
    xc = (x + w / 2) / W
    yc = (y + h / 2) / H
    wn = w / W
    hn = h / H
    return (xc, yc, wn, hn)


def union_bbox(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    x_min = min(x1, x2)  # top-left x
    y_min = min(y1, y2)  # top-left y
    x_max = max(x1 + w1, x2 + w2)  # bottom-right x
    y_max = max(y1 + h1, y2 + h2)  # bottom-right y
    return (x_min, y_min, x_max - x_min, y_max - y_min)

BALL_CLASS_ID = 0


SPLITS = ["train", "val"]

for split in SPLITS:
    IMG_DIR = os.path.join(r"D:\dataset_new_split", "images", split)
    LBL_DIR = os.path.join(r"D:\dataset_new_split", "labels", split)
    img_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png"))])

    print(f"\nProcessing split: {split} — {len(img_files)} images found")

    for i in range(len(img_files) - 1):
        img_name1 = img_files[i]
        img_name2 = img_files[i + 1]

        stem1 = os.path.splitext(img_name1)[0]
        stem2 = os.path.splitext(img_name2)[0]

        clip1 = stem1.split("__")[0]  # "Football_19_PSG_FCB_FG_dataset"
        clip2 = stem2.split("__")[0]  # "Football_20_HOF_FRA_FG_dataset"
        if clip1 != clip2:            # skip first and last images from different matches
            continue

        img1 = cv2.imread(os.path.join(IMG_DIR, img_name1))
        img2 = cv2.imread(os.path.join(IMG_DIR, img_name2))

        lab1 = read_yolo_label(os.path.join(LBL_DIR, stem1 + ".txt"))
        lab2 = read_yolo_label(os.path.join(LBL_DIR, stem2 + ".txt"))

        gray1_u8 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2_u8 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        gray1 = gray1_u8.astype(np.float32)
        gray2 = gray2_u8.astype(np.float32)

        dx, dy = 0.0, 0.0
        M = None

        pts1 = cv2.goodFeaturesToTrack(
            gray1_u8,  # expects 8 bit input
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30
        )

        if pts1 is not None:
            pts2, status, err = cv2.calcOpticalFlowPyrLK(
                gray1_u8, gray2_u8, pts1, None
            )

            status = status.reshape(-1)  # (N, 1) -> (N,)
            good1 = pts1[status == 1]  # only tracked points
            good2 = pts2[status == 1]  # only tracked points in frame 2

            good1_xy = good1.reshape(-1, 2)  # (N, 1, 2) -> # (N, 2)
            good2_xy = good2.reshape(-1, 2)

            if len(good1_xy) > 20:  # find the best transformation to map the points from frame 2 back to frame 1
                M, inliers = cv2.estimateAffinePartial2D(
                    good2_xy, good1_xy,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,  # a point counts as inlier
                    maxIters=2000,  # if after transforming it lands
                    confidence=0.99  # 3 pixels of the target
                )
                # M is 2x3 matrix like
                # [ a  b  tx ]
                # [ c  d  ty ]
                # inliers is array of 0/1 values teling what are good points

                if M is not None:
                    dx = float(M[0, 2])  # translation in x tx
                    dy = float(M[1, 2])  # translation in y ty

                    img2 = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[
                        0]))  # apply transform to the picture, cut the last channels dimension from img2
                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(
                        np.float32)  # recompute grayscale, convert back to float32

        A = gray1 - gray2  # (-255...255)
        B = (A + 255.0) / 2.0  # (0...255)
        gray_diff = np.clip(B, 0, 255).astype(np.uint8)  # force any values into the range (0-255)
        H, W = gray_diff.shape

        cls1, xc1, yc1, wn1, hn1 = lab1
        cls2, xc2, yc2, wn2, hn2 = lab2

        b1 = yolo_to_xywh(xc1, yc1, wn1, hn1, W, H)
        b2 = yolo_to_xywh(xc2, yc2, wn2, hn2, W, H)

        if M is not None:
            b2_aligned = apply_affine_to_bbox_xywh(b2, M)
        else:
            b2_aligned = b2

        union = union_bbox(b1, b2_aligned)
        x, y, w, h = union
        x = max(0, x)  # ensure bbox is in the frame
        y = max(0, y)
        w = min(W - x, w)
        h = min(H - y, h)

        xc, yc, wn, hn = xywh_to_yolo(x, y, w, h, W, H)
        # Visualization
        bbox_vis = cv2.cvtColor(gray_diff, cv2.COLOR_GRAY2BGR)  # convert back to 3-channels to draw a bbox
        cv2.rectangle(bbox_vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)  # draw a bbox

        img_name = f"{stem1}.jpg"
        lbl_name = f"{stem1}.txt"

        cv2.imwrite(os.path.join(r"D:/dataset_diffstab/images", split, img_name), gray_diff)
        cv2.imwrite(os.path.join(r"D:/dataset_diffstab/bounding_box",split, img_name), bbox_vis)

        with open(os.path.join(r"D:/dataset_diffstab/labels", split, lbl_name), "w") as f:
            f.write(f"{BALL_CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

        print(f"{i + 1}/{len(img_files) - 1}")
