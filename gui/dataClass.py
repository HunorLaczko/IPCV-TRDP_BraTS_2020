from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import pydicom
import os
from os import path
import argparse
import fnmatch
import utils 


#### NetworkOutput Class
class NetworkOutput:
    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path
        self.output = utils.generate_prediction(input_path, model_path)
    
    def getSlice(self, label, axis, axis_index):
        if axis == 0 :
            slice = self.output[label, axis_index, :, :]
        elif axis == 1 :
            slice = self.output[label, :, axis_index, :]
        else :
            slice = self.output[label, :, :, axis_index]
        return slice


#### GroundTruth Class

class GroundTruth:
    def __init__(self, input_path):
        self.input_path = input_path
        self.fullGroundTruth = utils.get_ground_truth(input_path)
        self.axisIndex = -1
        self.axis = -1
        self.label = -1

    def getAxisIndex(self, axis, label):
        if self.axisIndex == -1 or axis != self.axis or label != self.label:
            self.axisIndex = self.determineAxisIndex(axis, label)
        return self.axisIndex
    
    def determineAxisIndex(self, axis, label):
        self.axis = axis
        self.label = label
        if axis == 0 : # x
            slice = self.fullGroundTruth[label,0,:,:]
            count = np.count_nonzero(self.fullGroundTruth[label,0,:,:]==1)
            for i in range(self.fullGroundTruth.shape[1]): 
                if np.count_nonzero(self.fullGroundTruth[label,i,:,:]==1) > count:
                    count = np.count_nonzero(self.fullGroundTruth[label,i,:,:]==1)
                    slice = self.fullGroundTruth[label,i,:,:]
                    ax = i

        elif axis == 1 : # y 
            slice = self.fullGroundTruth[label,:,0,:]
            count = np.count_nonzero(self.fullGroundTruth[label,:,0,:]==1)
            for i in range(self.fullGroundTruth.shape[2]): 
                if np.count_nonzero(self.fullGroundTruth[label,:,i,:]==1) > count:
                    count = np.count_nonzero(self.fullGroundTruth[label,:,i,:]==1)
                    slice = self.fullGroundTruth[label,:,i,:]
                    ax = i

        else :  # z
            slice = self.fullGroundTruth[label,:,:,0]
            count = np.count_nonzero(self.fullGroundTruth[label,:,:,0]==1)
            for i in range(self.fullGroundTruth.shape[3]): 
                if np.count_nonzero(self.fullGroundTruth[label,:,:,i]==1) > count:
                    count = np.count_nonzero(self.fullGroundTruth[label,:,:,i]==1)
                    slice = self.fullGroundTruth[label,:,:,i]
                    ax = i
        return ax
