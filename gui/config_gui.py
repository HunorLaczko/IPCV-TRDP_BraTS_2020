import os
import shutil

config = dict()
config["image_shape"] = (128, 128, 128)
config["labels"] = (1, 2, 4)
config["all_modalities"] = ["t1", "t1ce", "t2", "flair"]
config["training_modalities"] = config["all_modalities"]

config["output_dir"] = os.path.abspath("./output")
config["test_file"] = os.path.abspath("./data/test_ids.pkl")

config["data_file_test"] = os.path.abspath("./data/brats2018_data_test.h5")
config["test_file"] = os.path.abspath("./data/isensee_test_ids.pkl")

config["num_test_files"] = 1
config["model_file"] = ""
config["test_dir"] = ""




