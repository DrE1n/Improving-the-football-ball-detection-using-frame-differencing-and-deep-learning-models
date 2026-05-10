
import os
from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()  # compulsory using Windows

    model_type = "yolov10m"

    source_directory = r"D:\dataset_new_diffstab"

    data_cfg = os.path.join(source_directory, "dataset_stabilized_difference.yaml")

    override_cfg = os.path.join(source_directory, "augmentation_stabilized_difference.yaml")

    experiment_name = "stabilized_difference_run"

    epoch_amount = 100

    batch_size = 8

    image_size = 800

    selected_classes = [0]  # ball class

    model = YOLO(model_type)

    results = model.train(data = data_cfg, cfg = override_cfg, name = experiment_name, \
                      epochs = epoch_amount, imgsz = image_size, batch = batch_size, \
                       classes=selected_classes)