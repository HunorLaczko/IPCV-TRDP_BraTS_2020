import utils


# class storing the image data
class Data:
    def __init__(self, input_path):
        self.data_array = utils.get_data_array(input_path)
    
    def getData(self):
        return self.data_array

    # returns a given slice from the 3D data along the given axis
    def getSlice(self, label, axis, axis_index):
        if axis == 0:
            slice = self.data_array[label, axis_index, :, :]
        elif axis == 1:
            slice = self.data_array[label, :, axis_index, :]
        else:
            slice = self.data_array[label, :, :, axis_index]

        return slice
