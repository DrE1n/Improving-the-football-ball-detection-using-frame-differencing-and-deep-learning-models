Here are the files related to the practical part of the bachelor thesis development.

Structure:
1. Folder train_configs holds the configuration files used to train the model which provide the paths to the dataset and augmentation file. The augmentation files holds the data related to the training itself e.g. image size, batch size, number of epochs and settings related to the data augmentation applied to the image like rotation, scaling, translation, cropping, etc;
1. The other train_results folder holds the results of each model training in the form of graphs and confusion matrices;
2. Other .py files involve files used to apply the computer vision methods to the dataset(FrameDiff and DIffStab), train the model(train_yolo10), apply the model to the video footage(difstab_video and coloured video respectively) and files related to validate the trained model(val_sets_divide and val_yolo)



Changes to the existing files are possible