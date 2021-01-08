import sys
import os
import numpy as np
sys.path.append(os.path.abspath('../src'))
from unet3d.utils.utils import read_image
from config_gui import config


# numpy implementation of the dice calculation to avoid tensorflow dependency
def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    return np.mean(2. * (np.sum(y_true * y_pred, axis=axis) + smooth/2) /
                   (np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis) + smooth))


# it relies on code from the src folder and on the configuration,
# it creates a one-hot-encoded version of the image
def separate_labels(input):
    output = np.zeros((3,) + config["image_shape"])
    for i in range(3):
        output[i, :, :, :] = (input == config["labels"][i]).astype(dtype="uint8")

    return output


# it relies on code from the src folder and on the configuration,
# to avoid including that in multiple places, this wrapper function is created
def get_data_array(input_path):
    seg = read_image(input_path, config["image_shape"]).get_fdata()
    return separate_labels(seg)


