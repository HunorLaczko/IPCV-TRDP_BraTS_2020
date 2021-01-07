import sys
import os
import fnmatch
import shutil
import mega
import numpy as np
sys.path.append(os.path.abspath('../src'))
#import inference
from unet3d.utils.utils import read_image
from config_gui import config



def separate_labels(input):
    output = np.zeros((3,) + config["image_shape"])
    for i in range(3):
        output[i, :, :, :] = (input == config["labels"][i]).astype(dtype="uint8")

    return output


def get_slice(output, layer, axis, axis_index):
    #print("[DEBUG] getSlice for Ground Truth")
    try :
        if axis == 0 :
            slice = output[layer, axis_index, :, :]
        elif axis == 1 :
            slice = output[layer, :, axis_index, :]
        else :
            slice = output[layer, :, :, axis_index]
            
        return slice

    except : 
        print("Cannot get slice because : ")
        if len(output.shape) != 3:
            print("wrong output size")
        if axis<0 or axis>3:
            print("axis out of bound")
        elif output.shape[axis + 1]<axis_index:
            print("axis_index out of bound")


def get_data_array(input_path):
    seg = read_image(input_path, config["image_shape"]).get_fdata()
    return separate_labels(seg)


