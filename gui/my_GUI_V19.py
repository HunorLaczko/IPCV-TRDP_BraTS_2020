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
from viewerClass import Viewer
from dataClass import NetworkOutput, GroundTruth
import utils


class MainWindow(QMainWindow):

    ########################################################################################
    ######################################### INIT #########################################
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()

        # input path
        self.directory = ""


        # model paths 
        self.model_path_wavelossAttention = "./models/waveloss_attention.h5"
        self.model_path_waveloss = "./models/waveloss_no_attention.h5"
        self.model_path_baselineAttention = "./models/baseline_attention.h5"
        self.model_path_baseline = "./models/baseline_no_attention.h5"

        ########################################################################################
        ####################################### MAIN BOX #######################################
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.mainBox = QVBoxLayout()
        self.setWindowTitle("TRDP GUI : Vizualizing segmentation results")

        # How the main window is organized
        self.mainHorizontalBox = QHBoxLayout()
        self.leftBox = QVBoxLayout()
        self.rightBox = QVBoxLayout()
        self.explainationBox = QVBoxLayout()
        self.directorySelectionBox = QHBoxLayout()
        self.GTViewerBox = QVBoxLayout()
        self.attentionDisplayBox = QHBoxLayout()
        self.noAttentionDisplayBox = QHBoxLayout()

        
        # Alignement
        self.mainHorizontalBox.setAlignment(Qt.AlignTop)
        self.explainationBox.setAlignment(Qt.AlignTop)
        self.directorySelectionBox.setAlignment(Qt.AlignLeft)
        self.GTViewerBox.setAlignment(Qt.AlignTop)

        ####################################### MAIN BOX #######################################
        ########################################################################################
        

        ########################################################################################
        ################################### EXPLAINATION BOX ###################################
        introText = "This GUI allows you to vizualize segmentation results.\n\n"
        step1 = "To do so : \n\n1. Select a folder containing a single sample of BraTS20 inputs"
        step2 = "2. Select vizualization options."
        

        self.introLabel=QLabel()
        self.introLabel.setText(introText)
        self.introLabel.setAlignment(Qt.AlignLeft)

        self.step1Label=QLabel()
        self.step1Label.setText(step1)
        
        ## DIRECTORY SELECTION BOX 

        self.select_button = QPushButton("   Select directory   ")
        self.select_button.clicked.connect(self.getDirectory)
        self.directorySelectionBox.addWidget(self.select_button)
        # print name and path of the file
        self.dir_label=QLabel()
        self.directorySelectionBox.addWidget(self.dir_label)

        self.step2Label=QLabel()
        self.step2Label.setText(step2)


        ## OPTIONS ##
        self.optionSelectionBox = QHBoxLayout()
        self.optionSelectionBox.setAlignment(Qt.AlignLeft)
        self.outputLabelSelectionBox = QHBoxLayout()
        axisSelectionBox = QHBoxLayout()
        # Alignment 
        self.outputLabelSelectionBox.setAlignment(Qt.AlignLeft)
        axisSelectionBox.setAlignment(Qt.AlignLeft)

        # select label to display
        self.label_output = QLabel('   Label :  ') ### set a margin instead of spaces
        self.combobox_layer = QComboBox()
        self.combobox_layer.addItems([" Necrotic and non-enhancing tumor core ", " Peritumoral edema ", " GD-enhancing tumor "])
        
        
        self.combobox_layer.setCurrentIndex(0)
        self.combobox_layer.currentIndexChanged.connect(self.updateViews)
        self.outputLabelSelectionBox.addWidget(self.label_output)
        self.outputLabelSelectionBox.addWidget(self.combobox_layer)

        # Select axis to display
        self.label_axis = QLabel('   Axis :  ') ### set a margin instead of spaces
        self.combobox_axis = QComboBox()
        self.combobox_axis.addItems(["   x   ", "   y   ", "   z   "])
        self.combobox_axis.setCurrentIndex(0)
        self.combobox_axis.currentIndexChanged.connect(self.updateViews)
        axisSelectionBox.addWidget(self.label_axis)
        axisSelectionBox.addWidget(self.combobox_axis)

        # put it together in the optionSelectionBox
        
        self.optionSelectionBox.addWidget(self.step2Label)
        self.optionSelectionBox.addLayout(self.outputLabelSelectionBox)
        self.optionSelectionBox.addLayout(axisSelectionBox)

        # put it together in the explainationBox

        self.explainationBox.addWidget(self.introLabel)
        self.explainationBox.addWidget(self.step1Label)
        self.explainationBox.addLayout(self.directorySelectionBox)
        self.explainationBox.addLayout(self.optionSelectionBox)


        ################################### EXPLAINATION BOX ###################################
        ########################################################################################    


        ########################################################################################
        ################################ GT VIEWER in LeftBox ##################################

        
        self.GT_loaded_plot = Viewer(self)
        GTLabel=QLabel("Ground Truth")
        GTLabel.setAlignment(Qt.AlignCenter)
        self.GTViewerBox.addWidget(GTLabel)
        self.GTViewerBox.addWidget(self.GT_loaded_plot)

        
        ################################# GT VIEWER in LeftBox ##################################
        ########################################################################################


        ########################################################################################
        ################################## RESULT DISPLAY BOX ##################################

        # Show the result for each method

        # With Attention : inside attentionDisplayBox
        self.wavelossAttentionLayout = QVBoxLayout()
        self.wavelossAttentionLayout.setAlignment(Qt.AlignTop)
        # label
        wavelossAttentionLabel=QLabel("Waveloss with Attention")
        wavelossAttentionLabel.setAlignment(Qt.AlignCenter)
        self.wavelossAttentionLayout.addWidget(wavelossAttentionLabel)
        # Viewer
        self.wavelossAttention_loaded_plot = Viewer(self)
        self.wavelossAttentionLayout.addWidget(self.wavelossAttention_loaded_plot)
        self.attentionDisplayBox.addLayout(self.wavelossAttentionLayout)

        self.baselineAttentionLayout = QVBoxLayout()
        self.baselineAttentionLayout.setAlignment(Qt.AlignTop)
        # label
        wavelossLabel=QLabel("Waveloss without Attention")
        wavelossLabel.setAlignment(Qt.AlignCenter)
        self.baselineAttentionLayout.addWidget(wavelossLabel)
        # Viewer
        self.baselineAttention_loaded_plot = Viewer(self)
        self.baselineAttentionLayout.addWidget(self.baselineAttention_loaded_plot)
        self.attentionDisplayBox.addLayout(self.baselineAttentionLayout)

        # Without Attention : inside noAttentionDisplayBox

        self.wavelossLayout = QVBoxLayout()
        self.wavelossLayout.setAlignment(Qt.AlignTop)
        # label
        baselineAttentionLabel=QLabel("Baseline with Attention")
        baselineAttentionLabel.setAlignment(Qt.AlignCenter)
        self.wavelossLayout.addWidget(baselineAttentionLabel)
        # Viewer
        self.waveloss_loaded_plot = Viewer(self)
        self.wavelossLayout.addWidget(self.waveloss_loaded_plot)
        self.noAttentionDisplayBox.addLayout(self.wavelossLayout)

        self.baselineLayout = QVBoxLayout()
        self.baselineLayout.setAlignment(Qt.AlignTop)
        # label
        baselineLabel=QLabel("Baseline without Attention")
        baselineLabel.setAlignment(Qt.AlignCenter)
        self.baselineLayout.addWidget(baselineLabel)
        # Viewer
        self.baseline_loaded_plot = Viewer(self)
        self.baselineLayout.addWidget(self.baseline_loaded_plot)
        self.noAttentionDisplayBox.addLayout(self.baselineLayout)

        
        ################################## RESULT DISPLAY BOX ##################################
        ########################################################################################

        # Put everything in the mainBox
        
        #self.leftBox.addLayout(self.directorySelectionBox)
        self.leftBox.addLayout(self.GTViewerBox)
        self.rightBox.addLayout(self.attentionDisplayBox)
        self.rightBox.addLayout(self.noAttentionDisplayBox)

        self.mainHorizontalBox.addLayout(self.leftBox, stretch=1)
        self.mainHorizontalBox.addLayout(self.rightBox, stretch=2)
        
        self.mainBox.addLayout(self.explainationBox)
        self.mainBox.addLayout(self.mainHorizontalBox)
        self.centralWidget.setLayout(self.mainBox)
    
    
    ######################################### INIT #########################################
    ########################################################################################



    ########################################################################################
    #################################### TOOL FUNCTIONS ####################################
    def is_correct_directory(self, directory):
        # check that the files are there
        # check that there are the exact amount of files
        # check that the name of the files correspond to the name of the directory  (same patient)
        return bool(fnmatch.filter(os.listdir(directory), '*flair.nii.gz')) & bool(fnmatch.filter(os.listdir(directory), '*t1.nii.gz')) & bool(fnmatch.filter(os.listdir(directory), '*t1ce.nii.gz')) & bool(fnmatch.filter(os.listdir(directory), '*t2.nii.gz'))
    
    def getDirectory(self):
        try:
            directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            if self.is_correct_directory(directory):
                self.directory = directory
                self.dir_label.setText(directory)
                self.dir_label.setStyleSheet('color: blue')
                # get ouputs
                self.getOuputs()
                self.updateViews()

            else :
                self.dir_label.setText("This directory should only contain 5 files, of form :\n'*flair.nii.gz', '*t1.nii.gz', '*t1ce.nii.gz', '*t2.nii.gz' and  '*seg.nii.gz'! ")
                self.dir_label.setStyleSheet('color: red')

        except:
            self.dir_label.setText("")
            #pass
    
    
    def getOuputs(self):
        print("[DEBUG] Hello from getOutputs method")
        self.groundTruth = GroundTruth(self.directory)
        self.wavelossAttentionOutput = NetworkOutput(self.directory, self.model_path_wavelossAttention)
        self.wavelossOutput = NetworkOutput(self.directory, self.model_path_waveloss)
        self.baselineAttentionOutput = NetworkOutput(self.directory, self.model_path_baselineAttention)
        self.baselineOutput = NetworkOutput(self.directory, self.model_path_baseline)
    
    def updateViews(self):
        print("[DEBUG] Hello from updateViews method")
        axis = self.combobox_axis.currentIndex()
        layer = self.combobox_layer.currentIndex()
        axis_index = self.groundTruth.getAxisIndex(axis, layer)
        self.GT_loaded_plot.plot(utils.getSlice(self.groundTruth, layer, axis, axis_index))
        self.wavelossAttention_loaded_plot.display(self.wavelossAttentionOutput.getSlice(layer, axis, axis_index))
        self.baselineAttention_loaded_plot.display(self.baselineAttentionOutput.getSlice(layer, axis, axis_index))
        self.waveloss_loaded_plot.display(self.wavelossOutput.getSlice(layer, axis, axis_index))
        self.baseline_loaded_plot.display (self.baselineOutput.getSlice(layer, axis, axis_index))

#################################### TOOL FUNCTIONS ####################################
########################################################################################



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow(app)
    ex.show()
    sys.exit(app.exec_( ))








