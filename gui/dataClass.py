import numpy as np
import utils
import numpy as np

import utils



class Data:
    def __init__(self, input_path):
        self.data_array = utils.get_data_array(input_path)
    
    def getArray(self) :
        return self.data_array
    
    def getSlice(self, label, axis, axis_index):
        return utils.get_slice(self.data_array, label, axis, axis_index)