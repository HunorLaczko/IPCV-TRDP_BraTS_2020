import os
import datetime

config = dict()
config["experiment_name"] = "final_waveloss_3_no_attention_"
config["base_folder"] = "/net/cremi/hlaczko/espaces/travail/"

config["image_shape"] = (128, 128, 128)         # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = None                    # switch to None to train on the whole image
config["labels"] = (1, 2, 4)                    # the label numbers on the input image
config["n_base_filters"] = 16                   # number of starting features
config["n_labels"] = len(config["labels"])
config["depth"] = 7                             # the depth of the network
config["n_segmentation_levels"] = 6             # number of segmentation levels created
config["dropout_rate"] = 0.3                    # dropout rate in the middle of convolutional blocks
config["deconvolution"] = True                  # if False, will use upsampling instead of deconvolution
config["deconvolution_last"] = True             # if False, will use upsampling instead of deconvolution
config["use_attention"] = False                 # wether to use the CBAM at the encoding stage
config["attention_ratio"] = 16                  # the reduction ratio for the CBAM

config["all_modalities"] = ["t1", "t1ce", "t2", "flair"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]

config["batch_size"] = 1                    # batch size used, most function only work with batch size of 1
config["validation_batch_size"] = 1
config["n_epochs"] = 300                    # cutoff the training after this many epochs, default 500
config["patience"] = 30                     # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 120                  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 1e-4      # initial learning rate
config["learning_rate_drop"] = 0.1          # factor by which the learning rate will be reduced
config["validation_split"] = 0.9            # portion of the data that will be used for training
config["flip"] = False                      # augments the data by randomly flipping an axis during
config["permute"] = False                   # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = 0.25                    # switch to None if you want no distortion
config["augment"] = True                    # config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0      # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True                 # if True, then patches without any target will be skipped

config["img_dir"] = config["base_folder"] + "data/MICCAI_BraTS2020_TrainingData"
config["label_dir"] = config["base_folder"] + "data/MICCAI_BraTS2020_TrainingData"
config["test_dir"] = config["base_folder"] + "data/MICCAI_BraTS2020_ValidationData" # or change 'ValidationData' --> 'TestData' when you predict for test data
config["num_test_files"] = 125              # Currently, this is number of validation files, change it to number of test files when you predict for test files

config["data_file"] = config["base_folder"] + "model/brats20_data.h5"
config["data_file_test"] = config["base_folder"] + "model/brats20_data_test.h5"
config["model_file"] = config["base_folder"] + "model/isensee_2017_model_" + config["experiment_name"] + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".h5"

config["training_file"] = config["base_folder"] + "data/model/training_ids.pkl"
config["validation_file"] = config["base_folder"] + "data/model/validation_ids.pkl"
config["test_file"] = config["base_folder"] + "data/model/test_ids.pkl"

config["use_tensorboard"] = True        # specify if tensorboard logging is wanted
config["tensorboard_logdir"] = config["base_folder"] + "logs/" + config["experiment_name"] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

config["overwrite"] = False             # If True, will previous files. If False, will use previously written files.

config["waveloss_use"] = True           # Use waveloss as a loss function, otherwise use weighted dice loss
config["waveloss_labelwise"] = False    # Perform waveloss seperately for each label and add them together
config["waveloss_spaw_log"] = True      # Use logarithmically increasing spatial weights for waloss, otherwise use linearly increasing
config["waveloss_valw_log"] = True      # Use logarithmically increasing value weights for waloss, otherwise use linearly increasing
config["waveloss_spainc"] = 5           # Spatial increase step size
config["waveloss_valinc"] = 0.1         # Value increase step size
config["waveloss_num_steps"] = 10       # Number of iterations for one comparison

