import os
import cv2
import shutil
import pandas as pd


DATASET_ROOT = "D:\dataset_new"          # raw dataset with clip folders
CSV_DIR      = os.path.join(DATASET_ROOT, "csv")

SPLIT_DATASETS = {
    "base": "D:\dataset_new_split",
    #"diff":     "D:\dataset_new_diff",
    "diffstab":  "D:\dataset_new_diffstab",
}
OUTPUT_ROOT = "D:\dataset_attribute_val2"  # where attribute val sets will be saved

attribute_map = {}
for csv_file in sorted(os.listdir(CSV_DIR)):
    csv_stem    = os.path.splitext(csv_file)[0]  # "Football_19_PSG_FCB_FG"
    folder_name = csv_stem + "_dataset"          # "Football_19_PSG_FCB_FG_dataset"
    df = pd.read_csv(os.path.join(CSV_DIR, csv_file)) # read .csv into a dataframe
    for _, row in df.iterrows():
        raw_filename = str(row["filename"])   # "000030_Football_19_PSG_FCB_FG.png"
        attribute    = str(row.get("attribute", "unknown")).strip().lower()
        attribute_map[(folder_name, raw_filename)] = attribute # append the dictionary


all_attributes = sorted(set(attribute_map.values()))
print(f"Attributes found: {all_attributes}") # check the attributes


for dataset_name, dataset_root in SPLIT_DATASETS.items():
    val_img_dir = os.path.join(dataset_root, "images", "val") # val image folder
    val_lbl_dir = os.path.join(dataset_root, "labels", "val") # val labels folder

    print(f"\nProcessing dataset: {dataset_name}") # keep an eye at the current set

    for attr in all_attributes:   # make a output directory
        for subdir in ("images", "labels"):
            os.makedirs(os.path.join(OUTPUT_ROOT, dataset_name, attr, subdir), exist_ok=True)

    matched   = 0
    unmatched = 0
    for img_file in sorted(os.listdir(val_img_dir)):
        stem = os.path.splitext(img_file)[0]  # e.g. "Football_19_PSG_FCB_FG_dataset__000030_Football_19_PSG_FCB_FG"
        parts       = stem.split("__", 1) # split on the double underscore to recover original name
        folder_name = parts[0]                # "Football_19_PSG_FCB_FG_dataset"
        bare_stem   = parts[1]                # "000030_Football_19_PSG_FCB_FG"

        attribute = None

        for ext in (".png", ".jpg"):  # extensions for original and processed datasets
            key = (folder_name, bare_stem + ext)
            if key in attribute_map:
                attribute = attribute_map[key] # add attribute
                break

        if attribute is None:
            unmatched += 1
            continue

        src_img = os.path.join(val_img_dir, img_file)  # source path for image
        src_lbl = os.path.join(val_lbl_dir, stem + ".txt") # source path for respective YOLO label

        dst_img = os.path.join(OUTPUT_ROOT, dataset_name, attribute, "images", img_file)    # destination path for attribute image
        dst_lbl = os.path.join(OUTPUT_ROOT, dataset_name, attribute, "labels", stem + ".txt") # destination path for attribute label
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)
        matched += 1

    print(f"  {matched} matched val, {unmatched} unmatched/skipped")
    for attr in all_attributes:   # prints summary
        attr_img_dir = os.path.join(OUTPUT_ROOT, dataset_name, attr, "images")
        count = len(os.listdir(attr_img_dir)) if os.path.exists(attr_img_dir) else 0
        print(f"    [{attr}]: {count} images")

