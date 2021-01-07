import sys
import os
import platform

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import utils
from dataClass import Data
from viewerClass import Viewer


class MainWindow(QMainWindow):

    ########################################################################################
    ######################################### INIT #########################################
    
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()

        # input path
        self.directory = "examples"
        self.result_directory = "result"
        self.truth_directory = "truth"
        self.prefix = "BraTS20_Training_"
        self.result_end = ".nii.gz"
        self.gt_end = "_seg.nii.gz"

        ########################################################################################
        ####################################### MAIN BOX #######################################
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.mainBox = QVBoxLayout()
        self.setWindowTitle("TRDP GUI : Visualizing segmentation results")

        # How the main window is organized
        self.mainHorizontalBox = QHBoxLayout()
        self.leftBox = QVBoxLayout()
        self.rightBox = QVBoxLayout()
        self.explanationBox = QVBoxLayout()
        self.directorySelectionBox = QHBoxLayout()
        self.GTViewerBox = QVBoxLayout()
        self.attentionDisplayBox = QHBoxLayout()
        self.noAttentionDisplayBox = QHBoxLayout()

        # Alignment
        self.mainHorizontalBox.setAlignment(Qt.AlignTop)
        self.explanationBox.setAlignment(Qt.AlignTop)
        self.directorySelectionBox.setAlignment(Qt.AlignLeft)
        self.GTViewerBox.setAlignment(Qt.AlignTop)

        ####################################### MAIN BOX #######################################
        ########################################################################################
        

        ########################################################################################
        ############################# EXPLAINATIONAND OPTIONS BOX ##############################
        introText = "This GUI allows for visualizing segmentation results with four different methods.\n"
        step1 = "1. Select a patient folder"
        step2 = "2. Select visualization options."
        

        self.introLabel=QLabel()
        self.introLabel.setText(introText)
        self.introLabel.setAlignment(Qt.AlignCenter)

        self.step1Label=QLabel()
        self.step1Label.setText(step1)


        ## DIRECTORY SELECTION BOX 

        self.combobox_patient = QComboBox()
        self.combobox_patient.addItems(["   Select a patient   ", "001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015"])
        self.patient_index = -1
        self.combobox_patient.setCurrentIndex(0) # if zero, then nothing
        self.combobox_patient.currentIndexChanged.connect(self.updatePatient)
        self.directorySelectionBox.addWidget(self.combobox_patient)

        ## OPTIONS ##

        self.step2Label=QLabel()
        self.step2Label.setText(step2)

        self.optionSelectionBox = QHBoxLayout()
        self.optionSelectionBox.setAlignment(Qt.AlignLeft)
        self.labelSelectionBox = QHBoxLayout()
        self.axisSelectionBox = QHBoxLayout()
        self.sliceSelectionBox = QHBoxLayout()
        # Alignment 
        self.labelSelectionBox.setAlignment(Qt.AlignLeft)
        self.axisSelectionBox.setAlignment(Qt.AlignLeft)

        # select label to display
        self.label_output = QLabel('   Label :  ') ### set a margin instead of spaces
        self.combobox_layer = QComboBox()
        #self.combobox_layer.addItems([" Necrotic and non-enhancing tumor core ", " Peritumoral edema ", " GD-enhancing tumor "])
        self.combobox_layer.addItems([" ET - enhanced tumor ", " WT - whole tumor ", " TC - tumor core "])
        
        self.combobox_layer.setCurrentIndex(0)
        self.combobox_layer.currentIndexChanged.connect(self.updateViews)
        self.labelSelectionBox.addWidget(self.label_output)
        self.labelSelectionBox.addWidget(self.combobox_layer)

        # Select axis to display
        self.label_axis = QLabel('   Axis :  ')  # set a margin instead of spaces
        self.combobox_axis = QComboBox()
        self.combobox_axis.addItems(["   x   ", "   y   ", "   z   "])
        self.combobox_axis.setCurrentIndex(0)
        self.combobox_axis.currentIndexChanged.connect(self.updateViews)
        self.axisSelectionBox.addWidget(self.label_axis)
        self.axisSelectionBox.addWidget(self.combobox_axis)

        # Select Slice
        self.label_cursor = QLabel('   Slice :  ')
        self.slideCursor = QSlider(Qt.Horizontal)

        self.slideCursor.setMinimum(0)
        self.slideCursor.setMaximum(127)
        self.slideCursor.setSingleStep(1)
        self.slideCursor.setTickInterval(10)
        self.slideCursor.setTickPosition(QSlider.TicksBelow)

        self.slideCursor.setValue(65)
        self.sliceCursorLabel = QLabel(str(self.slideCursor.value()))

        self.slideCursor.valueChanged.connect(self.updateSliceIndex)

        self.sliceSelectionBox.addWidget(self.label_cursor)
        self.sliceSelectionBox.addWidget(self.slideCursor)
        self.sliceSelectionBox.addWidget(self.sliceCursorLabel)

        # put it together in the optionSelectionBox
        self.optionSelectionBox.addLayout(self.labelSelectionBox)
        self.optionSelectionBox.addLayout(self.axisSelectionBox)
        self.optionSelectionBox.addLayout(self.sliceSelectionBox)

        # put it together in the explanationBox
        self.explanationBox.addWidget(self.introLabel)
        self.explanationBox.addWidget(self.step1Label)
        self.explanationBox.addLayout(self.directorySelectionBox)
        self.explanationBox.addWidget(self.step2Label)
        self.explanationBox.addLayout(self.optionSelectionBox)


        ############################# EXPLAINATIONAND OPTIONS BOX ##############################
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

        ##### With Attention : inside attentionDisplayBox

        ### Waveloss with attention
        self.wavelossAttentionLayout = QVBoxLayout()
        self.wavelossAttentionLayout.setAlignment(Qt.AlignTop)
        # label
        wavelossAttentionLabel=QLabel("Waveloss with Attention")
        wavelossAttentionLabel.setAlignment(Qt.AlignCenter)
        self.wavelossAttentionLayout.addWidget(wavelossAttentionLabel)
        # Viewer
        self.wavelossAttention_loaded_plot = Viewer(self)
        self.wavelossAttentionLayout.addWidget(self.wavelossAttention_loaded_plot)
        #self.wavelossAttentionLayout.addSpacing(5)
        # dice label
        self.wavelossAttentionDICELabel = QLabel("")
        self.wavelossAttentionDICELabel.setAlignment(Qt.AlignCenter)
        self.wavelossAttentionLayout.addWidget(self.wavelossAttentionDICELabel)
        self.attentionDisplayBox.addLayout(self.wavelossAttentionLayout)
        

        ### Baseline with attention
        self.baselineAttentionLayout = QVBoxLayout()
        self.baselineAttentionLayout.setAlignment(Qt.AlignTop)
        # label
        wavelossLabel = QLabel("Baseline with Attention")
        wavelossLabel.setAlignment(Qt.AlignCenter)
        self.baselineAttentionLayout.addWidget(wavelossLabel)
        # Viewer
        self.baselineAttention_loaded_plot = Viewer(self)
        self.baselineAttentionLayout.addWidget(self.baselineAttention_loaded_plot)
        # dice label
        self.wavelossDICELabel = QLabel("")
        self.wavelossDICELabel.setAlignment(Qt.AlignCenter)
        self.baselineAttentionLayout.addWidget(self.wavelossDICELabel)
        self.attentionDisplayBox.addLayout(self.baselineAttentionLayout)
        

        ##### Without Attention : inside noAttentionDisplayBox

        ### Waveloss
        self.wavelossLayout = QVBoxLayout()
        self.wavelossLayout.setAlignment(Qt.AlignTop)
        # label
        baselineAttentionLabel = QLabel("Waveloss without Attention")
        baselineAttentionLabel.setAlignment(Qt.AlignCenter)
        self.wavelossLayout.addWidget(baselineAttentionLabel)
        # Viewer
        self.waveloss_loaded_plot = Viewer(self)
        self.wavelossLayout.addWidget(self.waveloss_loaded_plot)
        # dice label
        self.baselineAttentionDICELabel = QLabel("")
        self.baselineAttentionDICELabel.setAlignment(Qt.AlignCenter)
        self.wavelossLayout.addWidget(self.baselineAttentionDICELabel)  
        self.noAttentionDisplayBox.addLayout(self.wavelossLayout)
        

        ### Baseline
        self.baselineLayout = QVBoxLayout()
        self.baselineLayout.setAlignment(Qt.AlignTop)
        # label
        baselineLabel = QLabel("Baseline without Attention")
        baselineLabel.setAlignment(Qt.AlignCenter)
        self.baselineLayout.addWidget(baselineLabel)
        # Viewer
        self.baseline_loaded_plot = Viewer(self)
        self.baselineLayout.addWidget(self.baseline_loaded_plot)
        # dice label
        self.baselineDICELabel = QLabel("")
        self.baselineDICELabel.setAlignment(Qt.AlignCenter)
        self.baselineLayout.addWidget(self.baselineDICELabel)
        self.noAttentionDisplayBox.addLayout(self.baselineLayout)
        
         
        ################################## RESULT DISPLAY BOX ##################################
        ########################################################################################

        # Put everything in the mainBox
        self.leftBox.addLayout(self.GTViewerBox)
        self.rightBox.addLayout(self.attentionDisplayBox)
        self.rightBox.addLayout(self.noAttentionDisplayBox)

        self.mainHorizontalBox.addLayout(self.leftBox, stretch=1)
        self.mainHorizontalBox.addLayout(self.rightBox, stretch=2)
        
        self.mainBox.addLayout(self.explanationBox)
        self.mainBox.addLayout(self.mainHorizontalBox)
        self.centralWidget.setLayout(self.mainBox)

        # disable resizing
        # needs different size for different platforms because of PyQt has different behaviour
        if platform.system() == 'Windows':
            self.setFixedSize(QSize(900, 820))
        else:
            self.setFixedSize(QSize(900, 900))
    
    ######################################### INIT #########################################
    ########################################################################################


    ########################################################################################
    #################################### TOOL FUNCTIONS ####################################


    def clearViews(self) : 
        #print("[DEBUG] clearViews")
        self.GT_loaded_plot.clearDisplay()
        self.wavelossAttention_loaded_plot.clearDisplay()
        self.baselineAttention_loaded_plot.clearDisplay()
        self.waveloss_loaded_plot.clearDisplay()
        self.baseline_loaded_plot.clearDisplay()

        self.wavelossAttentionDICELabel.setText("")
        self.wavelossDICELabel.setText("")
        self.baselineAttentionDICELabel.setText("")
        self.baselineDICELabel.setText("")

    def updateViews(self):
        #print("[DEBUG] updateViews")
        if self.patient_index == -1:  # If no patient is selected
            pass
        else:
            axis = self.combobox_axis.currentIndex()
            layer = self.combobox_layer.currentIndex()
            slice_index = self.slideCursor.value()
            
            self.GT_loaded_plot.display(self.groundTruth.getSlice(layer, axis, slice_index))
            self.wavelossAttention_loaded_plot.display(self.wavelossAttentionOutput.getSlice(layer, axis, slice_index))
            self.baselineAttention_loaded_plot.display(self.baselineAttentionOutput.getSlice(layer, axis, slice_index))
            self.waveloss_loaded_plot.display(self.wavelossOutput.getSlice(layer, axis, slice_index))
            self.baseline_loaded_plot.display(self.baselineOutput.getSlice(layer, axis, slice_index))
    
    def updatePatient(self):
        #print("[DEBUG] updatePatient")
        if self.patient_index + 1 == self.combobox_patient.currentIndex():
            pass
        else:
            self.patient_index = self.combobox_patient.currentIndex() - 1
            print(self.patient_index)
            if self.patient_index == -1:
                self.clearViews()
            else:
                self.groundTruth = Data(os.path.join(self.directory, self.truth_directory, self.prefix + self.combobox_patient.currentText() + self.gt_end))
                self.wavelossAttentionOutput = Data(os.path.join(self.directory, self.result_directory, "waveloss_attention", self.prefix + self.combobox_patient.currentText() + self.result_end))
                self.wavelossOutput = Data(os.path.join(self.directory, self.result_directory, "waveloss_no_attention", self.prefix + self.combobox_patient.currentText() + self.result_end))
                self.baselineAttentionOutput = Data(os.path.join(self.directory, self.result_directory, "baseline_attention", self.prefix + self.combobox_patient.currentText() + self.result_end))
                self.baselineOutput = Data(os.path.join(self.directory, self.result_directory, "baseline_no_attention", self.prefix + self.combobox_patient.currentText() + self.result_end))
                # compute Dice 
                y_true = self.groundTruth.getData()
                self.wavelossAttentionDICELabel.setText("dice : " + str(utils.weighted_dice_coefficient(y_true, self.wavelossAttentionOutput.getData())))
                self.wavelossDICELabel.setText("dice : " + str(utils.weighted_dice_coefficient(y_true, self.wavelossOutput.getData())))
                self.baselineAttentionDICELabel.setText("dice : " + str(utils.weighted_dice_coefficient(y_true, self.baselineAttentionOutput.getData())))
                self.baselineDICELabel.setText("dice : " + str(utils.weighted_dice_coefficient(y_true, self.baselineOutput.getData())))
                self.updateViews()

    def updateSliceIndex(self):
        #print("[DEBUG] updateSliceIndex")
        self.sliceCursorLabel.setText(str(self.slideCursor.value()))
        self.updateViews()

    #################################### TOOL FUNCTIONS ####################################
    ########################################################################################


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow(app)
    ex.show()
    sys.exit(app.exec_())








