from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import numpy as np
import mnist_neural_network as mnn
from PIL import Image

import MainWindow as MW

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = MW.Ui_Form()
        self.ui.setupUi(self)

        id = QFontDatabase.addApplicationFont('Roboto-Light.ttf')

        str = QFontDatabase.applicationFontFamilies(id)[0]

        font = QFont(str)

        font.setPixelSize(14)

        self.ui.label.setFont(font)
        self.ui.pushButton.setFont(font)

        self.painter = Painter()
        self.painter.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        self.ui.painterLayout.addWidget(self.painter)

        self.setMaximumSize(QSize(500, 550))
        self.setMinimumSize(QSize(500, 550))

        self.ui.pushButton.clicked.connect(self.painter.erase)

        self.classifier = mnn.ImgClassifier('numbers_data_11')

    def focusOutEvent(self, QFocusEvent):
        super().focusOutEvent(QFocusEvent)
        self.repaint()

    def paintEvent(self, QPaintEvent):
        super().paintEvent(QPaintEvent)
        img = self.painter.get_image()
        # print(img.reshape((28, 28)))
        if img.sum() != 0:
            pred = self.classifier.predicate(img)
            self.ui.label.setText('I see: ' + str(int(pred)))
        else:
            self.ui.label.setText('I see: empty')

class Painter(QLabel):
    def __init__(self):
        super(Painter, self).__init__()
        self.mouse = None
        self.img = QImage(QSize(500, 500), QImage.Format_Mono)

        self.erase()

    def focusOutEvent(self, QFocusEvent):
        super().focusOutEvent(QFocusEvent)
        self.repaint()

    def resizeEvent(self, *args, **kwargs):
        self.img = self.img.scaled(self.size(), Qt.KeepAspectRatio)

    def showEvent(self, *args, **kwargs):
        self.img = self.img.scaled(self.size(), Qt.KeepAspectRatio)


    def mousePressEvent(self, QMouseEvent):
        self.mouse = QMouseEvent.pos()
        # print(self.mouse)
        # self.update()

    def mouseMoveEvent(self, QMouseEvent):
        self.mouse = QMouseEvent.pos()
        self.repaint()

    def mouseReleaseEvent(self, QMouseEvent):
        self.mouse = None

    def paintEvent(self, QPaintEvent):
        super().paintEvent(QPaintEvent)

        if not self.mouse:
            return

        painter = QPainter(self)


        painterImg = QPainter(self.img)

        painterImg.setPen(QPen(Qt.black, 10))

        painterImg.drawEllipse(self.mouse, 5, 5)

        painter.drawImage(QPoint(0, 0), self.img)

    def erase(self):
        self.img = QImage(QSize(500, 500), QImage.Format_Mono)
        painterImg = QPainter(self.img)

        painterImg.setPen(QPen(Qt.white, 20))

        painterImg.fillRect(QRect(0, 0, 500, 500), Qt.white)
        self.repaint()

    def get_image(self):
        if self.img is None:
            return np.zeros((28 * 28))
        img = self.img.scaled(QSize(28, 28))
        img.save('test.png', 'PNG')



        arr = Image.open('test.png').convert('L')
        arr = np.array(arr)

        arr = arr.reshape((784))

        arr = abs(arr-255)

        width = img.width()
        height = img.height()

        # s = img.bits().asstring(width * height)
        # arr = np.fromstring(s, dtype=np.uint8).reshape((height * width))

        return arr


app = QApplication([])
window = MainWindow()
window.show()

sys.exit(app.exec())