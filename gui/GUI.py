import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import utils
from dataClass import NetworkOutput, GroundTruth
from viewerClass import Viewer


class MainWindow(QMainWindow):

    ########################################################################################
    ######################################### INIT #########################################
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()

        # input path
        self.directory = "./patient_results/"

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
        ############################# EXPLAINATIONAND OPTIONS BOX ##############################
        introText = "This GUI allows you to vizualize segmentation results with four different methods.\n"
        step1 = "To do so : \n\n1. Select a patient folder"
        step2 = "2. Select vizualization options."
        

        self.introLabel=QLabel()
        self.introLabel.setText(introText)
        self.introLabel.setAlignment(Qt.AlignLeft)

        self.step1Label=QLabel()
        self.step1Label.setText(step1)


        ## DIRECTORY SELECTION BOX 

        self.combobox_patient = QComboBox()
        self.combobox_patient.addItems(["   Select a patient   ", "", "", "", ""])
        self.patient_index = 0
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
        self.combobox_layer.addItems([" ET enhanced tumor ", " WT whole tumor ", " TC tumor core "])
        
        self.combobox_layer.setCurrentIndex(0)
        self.combobox_layer.currentIndexChanged.connect(self.updateViews)
        self.labelSelectionBox.addWidget(self.label_output)
        self.labelSelectionBox.addWidget(self.combobox_layer)

        # Select axis to display
        self.label_axis = QLabel('   Axis :  ') ### set a margin instead of spaces
        self.combobox_axis = QComboBox()
        self.combobox_axis.addItems(["   x   ", "   y   ", "   z   "])
        self.combobox_axis.setCurrentIndex(0)
        self.combobox_axis.currentIndexChanged.connect(self.updateViews)
        self.axisSelectionBox.addWidget(self.label_axis)
        self.axisSelectionBox.addWidget(self.combobox_axis)

        # Select Slice
        self.slideCursor = QSlider(Qt.Horizontal)

        self.slideCursor.setMinimum(0)
        self.slideCursor.setMaximum(127)
        self.slideCursor.setSingleStep(1)
        self.slideCursor.setTickInterval(10)
        self.slideCursor.setTickPosition(QSlider.TicksBelow)

        self.slideCursor.setValue(100)

        self.slice_index = self.slideCursor.value()
        self.slideCursorLabel = QLabel(str(self.slice_index))
    
        self.slideCursor.sliderPressed()
        #sliderReleased()


        self.sliceSelectionBox.addWidget(self.slideCursor)
        self.sliceSelectionBox.addWidget(self.slideCursorLabel)


        # put it together in the optionSelectionBox
        
        self.optionSelectionBox.addLayout(self.labelSelectionBox)
        self.optionSelectionBox.addLayout(self.axisSelectionBox)
        self.optionSelectionBox.addLayout(self.sliceSelectionBox)

        # put it together in the explainationBox

        self.explainationBox.addWidget(self.introLabel)
        self.explainationBox.addWidget(self.step1Label)
        self.explainationBox.addLayout(self.directorySelectionBox)
        self.explainationBox.addWidget(self.step2Label)
        self.explainationBox.addLayout(self.optionSelectionBox)


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

        ### With Attention : inside attentionDisplayBox

        ## Waveloss with attention
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

        ## Baseline with attention
        self.baselineAttentionLayout = QVBoxLayout()
        self.baselineAttentionLayout.setAlignment(Qt.AlignTop)
        # label
        wavelossLabel=QLabel("Baseline with Attention")
        wavelossLabel.setAlignment(Qt.AlignCenter)
        self.baselineAttentionLayout.addWidget(wavelossLabel)
        # Viewer
        self.baselineAttention_loaded_plot = Viewer(self)
        self.baselineAttentionLayout.addWidget(self.baselineAttention_loaded_plot)
        self.attentionDisplayBox.addLayout(self.baselineAttentionLayout)

        ### Without Attention : inside noAttentionDisplayBox

        ## Waveloss
        self.wavelossLayout = QVBoxLayout()
        self.wavelossLayout.setAlignment(Qt.AlignTop)
        # label
        baselineAttentionLabel=QLabel("Waveloss without Attention")
        baselineAttentionLabel.setAlignment(Qt.AlignCenter)
        self.wavelossLayout.addWidget(baselineAttentionLabel)
        # Viewer
        self.waveloss_loaded_plot = Viewer(self)
        self.wavelossLayout.addWidget(self.waveloss_loaded_plot)
        self.noAttentionDisplayBox.addLayout(self.wavelossLayout)

        ## Baseline
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
    
    def updatePatient(self): ## TODO : change this
        print("[DEBUG] Hello from updatePatient method")
        if self.patient_index == self.combobox_patient.currentIndex() :
            pass
        else : 
            self.patient_index = self.combobox_patient.currentIndex()
            if self.patient_index != 0 : 
                self.groundTruth = GroundTruth(self.directory)
                self.wavelossAttentionOutput = NetworkOutput(self.pred_path, self.model_path_wavelossAttention)
                self.wavelossOutput = NetworkOutput(self.pred_path, self.model_path_waveloss)
                self.baselineAttentionOutput = NetworkOutput(self.pred_path, self.model_path_baselineAttention)
                self.baselineOutput = NetworkOutput(self.pred_path, self.model_path_baseline)

            self.updateViews
        
    
    def updateViews(self): ## TODO : change this ?? 
        print("[DEBUG] Hello from updateViews method")
        if self.patient_index == 0 : # If no patient is selected
            self.GT_loaded_plot.clearCanvas()
            self.wavelossAttention_loaded_plot.clearCanvas()
            self.baselineAttention_loaded_plot.clearCanvas()
            self.waveloss_loaded_plot.clearCanvas()
            self.baseline_loaded_plot.clearCanvas()
        else :
            axis = self.combobox_axis.currentIndex()
            layer = self.combobox_layer.currentIndex()
            axis_index = self.groundTruth.getAxisIndex(axis, layer)
            self.GT_loaded_plot.display(utils.getSlice(self.groundTruth.getGroundTruth(), layer, axis, axis_index))
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








