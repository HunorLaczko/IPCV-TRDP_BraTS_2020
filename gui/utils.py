import sys
import os
import fnmatch
import shutil
import numpy as np
sys.path.append(os.path.abspath('../src'))
import inference
from unet3d.utils.utils import read_image
from config_gui import config



def create_util_folders():
    try:
        os.mkdir("data")
        os.mkdir("output")
    except:
        print("Failed to created necessary utility folders!")
        

def clean_up():
    try:
        shutil.rmtree("data", ignore_errors=True)
        shutil.rmtree("output", ignore_errors=True)
    except OSError as err:
        print("OS error: {0}!".format(err)) 
    except:
        print("Unknown error!")


def generate_prediction(input_path, model_path):
    create_util_folders()

    config["test_dir"] = input_path # has to be the parent folder in which only one patient folder can exist
    config["model_file"] = model_path
    
    inference.main(config)

    output_file = os.listdir("output")[0]
    segmentation = read_image(os.path.join("output", output_file), config["image_shape"]).get_data()
    output = np.zeros((3,) + config["image_shape"])
    for i in range(3):
        output[i,:,:,:] = (segmentation == config["labels"][i]).astype(dtype="uint8")

    clean_up()

    print("[DEBUG] generate_prediction worked")

    return output

def getSlice(output, label_index, axis, axis_index):
    try :
        if axis == 0 :
            slice = output[label_index, axis_index, :, :]
        elif axis == 1 :
            slice = output[label_index, :, axis_index, :]
        else :
            slice = output[label_index, :, :, axis_index]
            
        return slice

    except : 
        print("Cannot get slice because : ")
        if label_index>output.shape[0]:
            print("label_index out of bound")
        if len(output.shape) != 4:
            print("wrong output size")
        if axis<0 or axis>3:
            print("axis out of bound")
        elif output.shape[axis + 1]<axis_index:
            print("axis_index out of bound")

def get_ground_truth(input_path):
    file=fnmatch.filter(os.listdir(input_path), '*seg.nii.gz')
    gt = read_image(input_path + "/" + file[0], config["image_shape"]).get_fdata()
    return gt


#result = generate_prediction("/home/lachu/workspace/gui/IPCV-TRDP_BraTS_2020/gui/example", "/home/lachu/workspace/model/isensee_2017_model_final_baseline_2_attention__20201228-124617.h5")
#result = generate_prediction("/Users/anne-claire/Documents/github/IPCV-TRDP_BraTS_2020/gui/example", "./model_weights/baseline_no_attention.h5")

get_ground_truth("/Users/anne-claire/Documents/github/IPCV-TRDP_BraTS_2020/gui/example/BraTS20_Training_001")