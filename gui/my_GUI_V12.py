# python3 my_GUI_V9\ copy.py

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

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__()

        # Variables to change
        number_methods = 4

        ######################
        ###### MAIN BOX ######
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        self.mainBox = QVBoxLayout()
        self.setWindowTitle("Vizualize segmentation")

        # How the main window is organized
        self.mainHorizontalBox = QHBoxLayout()
        self.leftBox = QVBoxLayout()
        self.rightBox = QVBoxLayout()
        self.directorySelectionBox = QHBoxLayout()
        self.explainationBox = QVBoxLayout()
        self.directorySelectionBox = QHBoxLayout()
        self.optionsAndGTViewerBox = QHBoxLayout()
        self.attentionDisplayBox = QHBoxLayout()
        self.noAttentionDisplayBox = QHBoxLayout()

        # Alignement
        self.directorySelectionBox.setAlignment(Qt.AlignLeft)

        ##############################
        ###### EXPLAINATION BOX ######
        introText = "This GUI allows you to vizualize segmentation results.\n\n"
        step1 = "To do so : \n\n1. Select a folder containing a single sample of BraTS20 inputs"
        step2 = "2. Select vizualization options."
        
        self.introAndStep1Label=QLabel()
        self.introAndStep1Label.setText(introText+step1)
        
        ## DIRECTORY SELECTION BOX 

        self.select_button = QPushButton("   Select directory   ")
        self.select_button.clicked.connect(self.getDirectory)
        self.directorySelectionBox.addWidget(self.select_button)
        # print name and path of the file
        self.dir_label=QLabel()
        self.directorySelectionBox.addWidget(self.dir_label)

        self.step2Label=QLabel()
        self.step2Label.setText(step2)
        

        self.explainationBox.addWidget(self.introAndStep1Label)
        self.explainationBox.addLayout(self.directorySelectionBox)
        self.explainationBox.addWidget(self.step2Label)



        ###################################
        ###### optionsAndGTViewe BOX ######

        ## OPTIONS ##
        self.optionSelectionBox = QVBoxLayout()
        self.outputLabelSelectionBox = QHBoxLayout()
        self.axisSelectionBox = QHBoxLayout()
        # Alignment 
        self.outputLabelSelectionBox.setAlignment(Qt.AlignLeft)
        self.axisSelectionBox.setAlignment(Qt.AlignLeft)

        # select label to display
        self.label_output = QLabel('Display label : ') ### set a margin instead of spaces
        self.combobox_output = QComboBox()
        self.combobox_output.addItems(["1", "2", "3"])
        self.combobox_output.setCurrentIndex(0)
        self.combobox_output.currentIndexChanged.connect(self.outputChange)
        self.outputLabelSelectionBox.addWidget(self.label_output)
        self.outputLabelSelectionBox.addWidget(self.combobox_output)

        # Select axis to display
        self.label_axis = QLabel('Display axis : ') ### set a margin instead of spaces
        self.combobox_axis = QComboBox()
        self.combobox_axis.addItems(["x", "y", "z"])
        self.combobox_axis.setCurrentIndex(0)
        self.combobox_axis.currentIndexChanged.connect(self.axisChange)
        self.axisSelectionBox.addWidget(self.label_axis)
        self.axisSelectionBox.addWidget(self.combobox_axis)

        # put it together in the optionSelectionBox
        self.optionSelectionBox.addLayout(self.outputLabelSelectionBox)
        self.optionSelectionBox.addLayout(self.axisSelectionBox)

        ## GT VIEWER ##
        self.GTViewerBox = QVBoxLayout()
        loaded_plot = Viewer(self)
        loaded_plot.setMinimumHeight(250)
        loaded_plot.setMinimumWidth(250)
        self.GTViewerBox.addWidget(loaded_plot)

        ## PUT IT IN optionsAndGTViewerBox ##
        self.optionsAndGTViewerBox.addLayout(self.optionSelectionBox)
        self.optionsAndGTViewerBox.addLayout(self.GTViewerBox)


        ################################
        ###### RESULT DISPLAY BOX ######

        # Show the result for each method
        for i in range(number_methods//2):
            layout_plot = QVBoxLayout()
            loaded_plot = Viewer(self)
            loaded_plot.setMinimumHeight(250)
            loaded_plot.setMinimumWidth(250)
            layout_plot.addWidget(loaded_plot)
            self.attentionDisplayBox.addLayout(layout_plot)

        for i in range(number_methods//2):
            layout_plot = QVBoxLayout()
            loaded_plot = Viewer(self)
            loaded_plot.setMinimumHeight(250)
            loaded_plot.setMinimumWidth(250)
            layout_plot.addWidget(loaded_plot)
            self.noAttentionDisplayBox.addLayout(layout_plot)

        # Put everything in the mainBox

        
        self.leftBox.addLayout(self.directorySelectionBox)
        self.leftBox.addLayout(self.optionsAndGTViewerBox)
        self.rightBox.addLayout(self.attentionDisplayBox)
        self.rightBox.addLayout(self.noAttentionDisplayBox)

        self.mainHorizontalBox.addLayout(self.leftBox)
        self.mainHorizontalBox.addLayout(self.rightBox)
        
        self.mainBox.addLayout(self.explainationBox)
        self.mainBox.addLayout(self.mainHorizontalBox)
        self.centralWidget.setLayout(self.mainBox)


    def is_correct_directory(self, directory):
        return bool(fnmatch.filter(os.listdir(directory), '*flair.nii.gz')) & bool(fnmatch.filter(os.listdir(directory), '*t1.nii.gz')) & bool(fnmatch.filter(os.listdir(directory), '*t1ce.nii.gz')) & bool(fnmatch.filter(os.listdir(directory), '*t2.nii.gz'))
    
    def getDirectory(self):
        try:
            directory = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
            if self.is_correct_directory(directory):
                self.dir_label.setText(directory)
                self.dir_label.setStyleSheet('color: blue')
            else :
                self.dir_label.setText("This directory does not contain relevant files ! ")
                self.dir_label.setStyleSheet('color: red')
     
        except:
            self.dir_label.setText("")
            pass
    

    def axisChange(self):
        pass

    def outputChange(self):
        pass



# Class that shows the image
class Viewer(QGraphicsView):
    def __init__(self, parent=None):
        super(Viewer, self).__init__(parent)
        self.parent=parent
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHeightForWidth(True)
        self.setSizePolicy(sizePolicy) 
        #self.toolbar = NavigationToolbar(self.canvas, self)

        # self.canvas.setGeometry(0, 0, 1600, 500 )
        layout = QVBoxLayout()
        #layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Load the image read from the command line
        self.axes=self.figure.add_subplot(111)
        #img_path = args.image
        #self.dataset = pydicom.dcmread(img_path)

        # FOR METHOD 2:Generate and store 8bit representation of the image.
        #img = self.dataset.pixel_array.astype(float)

        ## Step 2. Rescaling grey scale between 0-255
        #img = img-img.min()
        #self.imgScaled = (np.maximum(img,0) / img.max()) * 255.0

        ## Step 3. Convert to uint8
        #self.imgScaled = np.uint8(self.imgScaled)

        self.max =  255
        self.min = 0

        #im = self.axes.imshow(self.imgScaled, cmap=plt.cm.bone, vmax = self.max, vmin=self.min)
        #self.figure.colorbar(im)
        self.axes.axis('off')
        #self.titleImage = img_path+'\n'
        #self.axes.set_title(self.titleImage)
        self.canvas.draw()
        self.canvas.show()   
         
      
    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow(app)
    ex.show()
    sys.exit(app.exec_( ))