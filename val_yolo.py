from ultralytics import YOLO
from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()  # compulsory using Windows
    model = YOLO(r"D:\dataset_new_runs\detect\base_run-2\weights\best.pt")
    results = model.val(
        #data  = r"D:\dataset_attribute_val\base\inhand\dataset.yaml",
        data  = r"D:\configs\dataset_colour.yaml",
        imgsz = 800,
        project = r"D:\dataset_new_eval\yolo_10_base2",
        name = "inhand"
    )
    print(f"mAP50-95: {results.box.map}")
