import cv2
import os
import pandas as pd

def parse_coordinates(coord_str):
    parts = str(coord_str).strip().strip('"').split(",")  # remove quotes and split by comma
    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])  # return as 4 floats

DATASET_ROOT = "D:/dataset_new"
CSV_DIR = os.path.join(DATASET_ROOT, "csv")
OUTPUT_ROOT = "D:/dataset_new_split"
label_map = {}  # label lookup dictionary

for csv_file in sorted(os.listdir(CSV_DIR)):
    if not csv_file.lower().endswith(".csv"):          # look for csv files
        continue
    csv_stem = os.path.splitext(csv_file)[0]        # extract the folder name "Football_19_PSG_FCB_FG"
    folder_name = csv_stem + "_dataset"
    df = pd.read_csv(os.path.join(CSV_DIR, csv_file)) # read the file
    for _, row in df.iterrows():
        raw_filename = str(row["filename"])            # get the respective image
        if str(row.get("type", "ball")).lower() != "ball":
            continue
        try:
            x_min, y_min, x_max, y_max = parse_coordinates(row["coordinates"])  # take bbox cordinates
        except Exception:
            continue

        label_map[folder_name][raw_filename] = (x_min, y_min, x_max, y_max)  # keep bbox


BALL_CLASS_ID = 0    # YOLO class index for ball
VAL_EVERY = 10   # train/val ratio

all_folders = sorted([
    d for d in os.listdir(DATASET_ROOT)
    if os.path.isdir(os.path.join(DATASET_ROOT, d))  # find image folders
    and d != "csv"
])

total_train = 0   # total images written to training set across all folders
total_val = 0   # total images written to validation set across all folders


for folder_name in all_folders:
    folder_path = os.path.join(DATASET_ROOT, folder_name)
    folder_labels = label_map[folder_name]                  # retrieve annotations for the image folder
    img_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png"))             # only include image files in the folder
    ])

    saved_train = 0
    saved_val = 0
    skipped  = 0

    for idx, img_name in enumerate(img_files):              # iterate over images with indexes
        if img_name not in folder_labels:                   # skip if no annotation exists
            skipped += 1
            continue

        img = cv2.imread(os.path.join(folder_path, img_name))  # read image into numpy array
        if img is None:                                         # skip if failed
            skipped += 1
            continue

        H, W = img.shape[:2]                               # extract height and width in pixels

        x_min, y_min, x_max, y_max = folder_labels[img_name]  # extract raw coordinates of the bbox

        #  convert pixel coordinates to YOLO normalized format
        xc = ((x_min + x_max) / 2) / W    # centre x: average of left and right edges, divided by image width
        yc = ((y_min + y_max) / 2) / H    # centre y: average of top and bottom edges, divided by image height
        wn = (x_max - x_min) / W          # normalized width: pixel width divided by image width
        hn = (y_max - y_min) / H          # normalized height: pixel height divided by image height

        stem  = os.path.splitext(img_name)[0]       # original filename without extension
        out_stem = f"{folder_name}__{stem}"         # combine folder name and filename with double underscore separator
        out_img_name = out_stem + ".jpg"            # output image filename
        out_lbl_name = out_stem + ".txt"            # output YOLO label filename


        split = "val" if (idx + 1) % VAL_EVERY == 0 else "train"     # assign to train or val split
        cv2.imwrite(os.path.join(r"D:\dataset_new_split\images", split, out_img_name), img)  # write to output
        with open(os.path.join(r"D:\dataset_new_split\"labels", split, out_lbl_name), "w") as f:    #class_id centre_x centre_y width height (all space-separated, normalized 0→1)
            f.write(f"{BALL_CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
        if split == "train":
            saved_train += 1
        else:
            saved_val += 1
    total_train += saved_train
    total_val += saved_val
    print(f"{folder_name}: {saved_train} train, {saved_val} val, {skipped} skipped")
print(f"\nTotal: {total_train} train / {total_val} val")

print(f"Actual split: {total_train/(total_train+total_val)*100:.1f}% train / "
      f"{total_val/(total_train+total_val)*100:.1f}% val")