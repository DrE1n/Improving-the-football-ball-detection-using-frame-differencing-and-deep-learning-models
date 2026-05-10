import cv2
import numpy as np
from ultralytics import YOLO

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
    imgsz         = 800,
    conf          = 0.05,
    ring_color    = (0, 255, 0),
    ring_radius   = 5,
    ring_thickness = 3,
):
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    model  = YOLO(model_path)

    ok, frame = cap.read()
    frame_idx = 0
    while True:
        if not ok:
            break
        frame_idx += 1

        # Run inference directly on the colour frame — no preprocessing
        result   = model.predict(source=frame, imgsz=imgsz, conf=conf, verbose=False)
        midpoint = get_bbox_center(result[0])

        vis = frame.copy()

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
        ok, frame = cap.read()

    cap.release()
    writer.release()
    print(f"Done. Saved to: {out_path}")

if __name__ == "__main__":
    generate_midpoint_video(
        video_path=r"C:\Users\andza\Downloads\output_SF_12ST_F95_FCN_1st_half_1min_020.mp4",
        out_path=r"D:\final_videos\colored_mAP45%.mp4",
        model_path=r"D:\dataset_new_runs\detect\base_run-2\weights\best.pt",
        imgsz=800,
        conf=0.05,
    )