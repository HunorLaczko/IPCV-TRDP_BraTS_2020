import utils


class Data:
    def __init__(self, input_path):
        self.data_array = utils.get_data_array(input_path)
    
    def getData(self):
        return self.data_array
    
    def getSlice(self, label, axis, axis_index):
        if axis == 0:
            slice = self.data_array[label, axis_index, :, :]
        elif axis == 1:
            slice = self.data_array[label, :, axis_index, :]
        else:
            slice = self.data_array[label, :, :, axis_index]

        return slice
