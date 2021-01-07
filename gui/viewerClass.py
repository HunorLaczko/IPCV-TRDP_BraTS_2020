import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas



# Class that shows the image
class Viewer(QGraphicsView):
    def __init__(self, parent=None):
        super(Viewer, self).__init__(parent)
        self.parent=parent
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.setMinimumHeight(250)
        self.setMinimumWidth(250)

        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHeightForWidth(True)
        self.setSizePolicy(sizePolicy) 

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.axes=self.figure.add_subplot(111)
        self.axes.axis('off')

    def display(self, slice) :
        print("[DEBUG] Viewer.display")
        self.axes.imshow(slice.T, cmap="gray", origin="lower")
        self.axes.axis('off')
        self.canvas.draw()
        self.canvas.show()  
    
    def clearCanvas(self) : 
        self.canvas.close()  
