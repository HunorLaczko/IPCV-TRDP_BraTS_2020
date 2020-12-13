import os
import datetime

config = dict()
config["experiment_name"] = "waveloss_labelwise_"
config["base_folder"] = "/autofs/unitytravail/travail/hlaczko/"

config["image_shape"] = (64, 64, 64)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = None  # switch to None to train on the whole image
config["labels"] = (1, 2, 4)  # the label numbers on the input image
config["n_base_filters"] = 16
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["t1", "t1ce", "t2", "flair"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

config["batch_size"] = 1
config["validation_batch_size"] = 1
config["n_epochs"] = 150  # cutoff the training after this many epochs, default 500
config["patience"] = 30  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 60  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 1e-4
config["learning_rate_drop"] = 0.1  # factor by which the learning rate will be reduced
config["validation_split"] = 0.9  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = False  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = 0.25  # switch to None if you want no distortion
config["augment"] = True # config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

# To be changed accordingly, if it is required.
config["img_dir"] = config["base_folder"] + "data/MICCAI_BraTS2020_TrainingData"
config["label_dir"] = config["base_folder"] + "data/MICCAI_BraTS2020_TrainingData"
config["test_dir"] = config["base_folder"] + "data/MICCAI_BraTS2020_ValidationData" # or change 'ValidationData' --> 'TestData' when you predict for test data
config["num_test_files"] = 125 # Currently, this is number of validation files, change it to number of test files when you predict for test files

config["data_file"] = "//home/lachu/workspace/model/brats20_data.h5"
config["data_file_test"] = config["base_folder"] + "model/brats20_data_test.h5"
config["model_file"] = config["base_folder"] + "model/isensee_2017_model_" + config["experiment_name"] + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"

config["training_file"] = config["base_folder"] + "data/model/training_ids.pkl"
config["validation_file"] = config["base_folder"] + "data/model/validation_ids.pkl"
config["test_file"] = config["base_folder"] + "data/model/test_ids.pkl"

config["use_tensorboard"] = True
config["tensorboard_logdir"] = config["base_folder"] + "logs/" + config["experiment_name"] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.
