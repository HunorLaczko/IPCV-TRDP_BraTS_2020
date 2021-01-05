import sys
import os
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
        pass
        

def clean_up():
    try:
        shutil.rmtree("data")
        shutil.rmtree("output")
    except:
        pass  


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

    return output

generate_prediction("/home/lachu/workspace/gui/data/data/training", "/home/lachu/workspace/model/isensee_2017_model_final_baseline_2_attention__20201228-124617.h5")