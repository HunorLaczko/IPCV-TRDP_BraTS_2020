import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap

VIEWER_SIZE = (275, 250)


# Class that shows the image
class Viewer(QGraphicsView):
    def __init__(self, parent=None):
        super(Viewer, self).__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setMinimumHeight(VIEWER_SIZE[0])
        self.setMinimumWidth(VIEWER_SIZE[1])

        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHeightForWidth(True)
        self.setSizePolicy(sizePolicy)

        # create a label that will show the image
        self.frame = QLabel()
        self.frame.setAlignment(Qt.AlignHCenter)
        self.frame.setMinimumHeight(250)
        self.frame.setMinimumWidth(250)
        self.frame.setSizePolicy(sizePolicy)

        # create layout
        layout = QHBoxLayout()
        layout.addWidget(self.frame)
        layout.addSpacing(5)
        self.setLayout(layout)

    # display given image, applies transformation from binary to uint8 and resize for the view size
    def display(self, slice):
        slice = (slice * 255).astype(dtype="uint8")
        pixmap = QPixmap.fromImage(QImage(slice.data, slice.shape[1], slice.shape[0], QImage.Format_Grayscale8))
        pixmap = pixmap.scaledToWidth(self.width() - 25)
        self.frame.setPixmap(pixmap)

    # sets an empty image
    def clearDisplay(self):
        self.display(np.zeros((128, 128, 128)))
