import sys
import os
import fnmatch
import shutil
import mega
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


def separate_labels(input):
    output = np.zeros((3,) + config["image_shape"])
    for i in range(3):
        output[i, :, :, :] = (input == config["labels"][i]).astype(dtype="uint8")

    return output


def generate_prediction(input_path, model_path):
    create_util_folders()

    config["test_dir"] = input_path # has to be the parent folder in which only one patient folder can exist
    config["model_file"] = model_path
    
    inference.main(config)

    output_file = os.listdir("output")[0]
    segmentation = read_image(os.path.join("output", output_file), config["image_shape"]).get_fdata()
    output = separate_labels(segmentation)

    clean_up()

    print("[DEBUG] generate_prediction worked")

    return output

def getSlice(output, layer, axis, axis_index):
    print("[DEBUG] getSlice for Ground Truth")
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

def get_ground_truth(input_path):
    file=fnmatch.filter(os.listdir(input_path), '*seg.nii.gz')
    seg = read_image(input_path + "/" + file[0], config["image_shape"]).get_fdata()
    return separate_labels(seg)


def download_models():
    models = ["baseline_attention", "baseline_no_attention", "waveloss_attention", "waveloss_no_attention"]
    model_urls = ["https://mega.nz/file/dZISgBxR#LBE8Hx6pORveE5E32556l2Rip1Z_aetmW139UoI4heA",
                  "https://mega.nz/file/ZFxwWRCY#nO-s8AoQtMbzPa0dG_imaxt-EjYswSdyshTLr8T8jMI",
                  "https://mega.nz/file/0FgAxDLS#kZK-44VOl4_wssH7XcjaikMfFlye-4X5dr8PM5kGVZ8",
                  "https://mega.nz/file/pB4SzRiQ#In6Vh3B5MhRWm5RUMWzB04STpnrXMXfWEqPz_7ytZgA"]

    m = mega.Mega()

    if not os.path.exists("./models"):
        os.mkdir("model_weights")

    for i in range(4):
        if not os.path.exists("./models/" + models[i] + ".h5"):
            try:
                m.download_url(model_urls[i])
            except:
                pass

            if not os.path.exists(models[i] + ".h5"):
                print("Failed to download model file!")

            shutil.move(models[i] + ".h5", "./models/" + models[i] + ".h5")


def set_up_directory(safe_for_input_dir, input_dir):
    files_to_remove = os.listdir(safe_for_input_dir)
    if len(files_to_remove) != 0 :
        for i in range (len(files_to_remove)):
            shutil.rmtree(os.path.join(safe_for_input_dir, files_to_remove[i]), ignore_errors=True)
    dir = shutil.copytree(input_dir, safe_for_input_dir)
    return dir

def check_correct_input_directory(input_directory_path):
    print("[ DEBUG ] utils.check_correct_input_directory ")
    contains_correct_files = bool(fnmatch.filter(os.listdir(input_directory_path), '*flair.nii.gz')) & bool(fnmatch.filter(os.listdir(input_directory_path), '*t1.nii.gz')) & bool(fnmatch.filter(os.listdir(input_directory_path), '*t1ce.nii.gz')) & bool(fnmatch.filter(os.listdir(input_directory_path), '*t2.nii.gz')) & bool(fnmatch.filter(os.listdir(input_directory_path), '*seg.nii.gz'))
    if not contains_correct_files :
        return False
    patient_number = -1
    try :
        patient_number = input_directory_path.split("/")[-1].split("_")[2]
    except : 
        return False
    list_files = sorted(os.listdir(input_directory_path))
    nb_files = len(list_files)
    if nb_files != 5 :
        print("there should be 5 files")
        return False
    for i in range (nb_files) :
        if list_files[i].split("_")[2] != patient_number : 
            print ("The files do not belong to the same patient ! ")
            return False
    return True


#result = generate_prediction("/home/lachu/workspace/gui/IPCV-TRDP_BraTS_2020/gui/example", "/home/lachu/workspace/model/isensee_2017_model_final_baseline_2_attention__20201228-124617.h5")
#result = generate_prediction("/Users/anne-claire/Documents/github/IPCV-TRDP_BraTS_2020/gui/example", "./model_weights/baseline_no_attention.h5")

#get_ground_truth("/Users/anne-claire/Documents/github/IPCV-TRDP_BraTS_2020/gui/example/BraTS20_Training_001")