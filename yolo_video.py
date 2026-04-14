import cv2
import numpy as np
from ultralytics import YOLO


def signed_diff(gray1, gray2):
    diff = gray1 - gray2
    signed = (diff + 255.0) / 2.0
    return np.clip(signed, 0, 255).astype(np.uint8)


def estimate_affine(gray1_u8, gray2_u8):

    pts1 = cv2.goodFeaturesToTrack(
        gray1_u8, # expects 8 bit input
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=30
    )

    if pts1 is not None:
        pts2, status, err = cv2.calcOpticalFlowPyrLK(
            gray1_u8, gray2_u8, pts1, None
        )

        status = status.reshape(-1) # (N, 1) -> (N,)
        good1 = pts1[status == 1]   # only tracked points
        good2 = pts2[status == 1]   # only tracked points in frame 2

        good1_xy = good1.reshape(-1, 2)  # (N, 1, 2) -> # (N, 2)
        good2_xy = good2.reshape(-1, 2)

        if len(good1_xy) > 20: # find the best transformation to map the points from frame 2 back to frame 1
            M, inliers = cv2.estimateAffinePartial2D(
                good2_xy, good1_xy,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0, # a point counts as inlier
                maxIters=2000,             # if after transforming it lands
                confidence=0.99            # 3 pixels of the target
            )
            # M is 2x3 matrix like
            # [ a  b  tx ]
            # [ c  d  ty ]
            # inliers is array of 0/1 values teling what are good points

    return M


def build_subtraction(img1, img2, use_stabilization=True):
    gray1_u8 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2_u8 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray1 = gray1_u8.astype(np.float32)

    if use_stabilization:  # can be false for subtraction only model
        M = estimate_affine(gray1_u8, gray2_u8)
        if M is not None:
            H, W = gray1_u8.shape
            img2_warp = cv2.warpAffine(img2, M, (W, H)) # apply transform to the picture, cut the last channels dimension from img2
            gray2 = cv2.cvtColor(img2_warp, cv2.COLOR_BGR2GRAY).astype(np.float32)  # recompute grayscale, convert back to float32
        else:
            gray2 = gray2_u8.astype(np.float32)
    else:
        gray2 = gray2_u8.astype(np.float32)

    return signed_diff(gray1, gray2)


def get_bbox_center(result):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    confs = result.boxes.conf.cpu().numpy()    # get all the conf scores
    best  = int(np.argmax(confs))              # and select the highest one

    xyxy  = result.boxes.xyxy[best].cpu().numpy()
    x1, y1, x2, y2 = xyxy     # convert bbox corners -> centre point
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx, cy)


def generate_midpoint_video(
    video_path,
    out_path,
    model_path,
    use_stabilization = True,
    imgsz             = 640,
    conf              = 0.05,
    out_size          = 640,
    ring_color        = (0, 255, 0),   # green ring
    ring_radius       = 5,
    ring_thickness    = 3,
):
    cap = cv2.VideoCapture(video_path)  # read the frame rate for correct speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec  = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, codec, fps, (W, H))

    model = YOLO(model_path)

    ok, frame1 = cap.read()  # read first two frames
    ok, frame2 = cap.read()

    subs12  = build_subtraction(frame1, frame2, use_stabilization)      # subtract frames, apply stabilisation
    result12 = model.predict(source=subs12, imgsz=imgsz, conf=conf, verbose=False)       # apply model on a stabilised subtraction
    bbox12   = get_bbox_center(result12[0])     # write down the bbox centre

    frame_idx = 0

    while True:
        ok, frame3 = cap.read()   # read the third frame
        frame_idx += 1

        # same processing as with first two frames
        subs23   = build_subtraction(frame2, frame3, use_stabilization)
        result23 = model.predict(source=subs23, imgsz=imgsz, conf=conf, verbose=False)
        bbox23   = get_bbox_center(result23[0])

        midpoint = None
        if bbox12 is not None and bbox23 is not None:
            midpoint = (                           # if there are detections
                (bbox12[0] + bbox23[0]) / 2.0,     # between frames, compute
                (bbox12[1] + bbox23[1]) / 2.0      # the midpoint  between them
            )
        elif bbox12 is not None:      # if one detection is missing,
            midpoint = bbox12         # the other one becomes midpoint
        elif bbox23 is not None:
            midpoint = bbox23

        vis = frame2.copy()

        if midpoint is not None:      # read the midpoint coords
            mx = int(midpoint[0])
            my = int(midpoint[1])
            mx = max(0, min(W - 1, mx))
            my = max(0, min(H - 1, my))

            cv2.circle(vis, (mx, my), ring_radius, ring_color, ring_thickness)
        writer.write(vis)

        if frame_idx % 10 == 0:
            status = f"({midpoint[0]:.0f}, {midpoint[1]:.0f})" if midpoint else "no detection"
            print(f"Frame {frame_idx:5d}  ball: {status}")

        frame1  = frame2     # sliding down the frames
        frame2  = frame3
        subs12  = subs23
        bbox12  = bbox23

    cap.release()
    writer.release()
    print(f"Done. Saved to: {out_path}")


if __name__ == "__main__":
    generate_midpoint_video(
        video_path        = r"C:\Users\andza\Downloads\output_SF_12ST_F95_FCN_1st_half_1min_020.mp4",
        out_path          = r"D:\outputs_streaming\midpoint_coloured.mp4",
        model_path        = r"C:\Users\andza\runs\detect\diffstab_new_split\weights\best.pt",
        use_stabilization = True,
        imgsz             = 640,
        conf              = 0.05,
        out_size          = 640,
    )