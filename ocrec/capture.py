import pyautogui as pag
from PIL import ImageGrab

from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtCore

from PyQt5.QtCore import *
from PyQt5.QtGui import *

from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QGridLayout, QWidget, QLabel, QMainWindow



# -*- coding: utf-8 -*-




class Screen(object):

    def __init__(self,captureWindow):
        #super().__init__(None, QtCore.Qt.WindowStaysOnTopHint)
        self.captureWindow = captureWindow
        print("Message from main window") 

    # def __init__(self):
    #     super().__init__(None, QtCore.Qt.WindowStaysOnTopHint)
    #     self.setupUi()
     

    def setupUi(self):

        
        self.x1 = None
        self.y1 = None
        
        self.x2 = None
        self.y2 = None
       

    
        self.centralwidget = QWidget(self.captureWindow)
        self.centralwidget.setObjectName("centralwidget")
    
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")

        self.captureWindow.setCursor(QCursor(Qt.CrossCursor))
        self.captureWindow.showMaximized()
        self.captureWindow.setWindowOpacity(0.1)



        self.screen = QLabel(self.centralwidget)
        #self.screen.setPixmap(QPixmap("screen.png"))
        self.screen.setScaledContents(True)
        self.screen.setObjectName("screen")

        
        self.gridLayout.addWidget(self.screen, 0, 0, 1, 1)
        self.captureWindow.setCentralWidget(self.centralwidget)
    
        #self.captureWindow.setMouseTracking(True)



    # 마우스 MOVE
    def mouseMoveEvent(self,event):
        self.draw_Line(event.x(),event.y())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.x1 = event.x()
            self.y1 = event.y()

    # 마우스 RELEASE
    def mouseReleaseEvent(self,event):  
        if event.button() == Qt.LeftButton:
            #self.draw_Line(event.x(),event.y())
            self.x2 = event.x()
            self.y2 = event.y()
            self.screenshot()

    def draw_Line(self,x,y):
        
        if self.x1 is None:
            self.x1 = x
            self.y1 = y
        else:
            self.x2 = x
            self.y2 = y

            self.img = QPixmap("screen.png")
            painter = QtGui.QPainter(self.img)
            painter.setPen(QPen(Qt.red, 5, Qt.SolidLine))
            painter.drawRect(self.x1,self.y1,self.x2,self.y2)
            painter.end()
            self.screen.setPixmap(QtGui.QPixmap(self.img))

    def screenshot(self):
        capture_width = self.x2 - self.x1
        capture_height = self.y2 - self.y1
        path = r"C:\kyun\conda\ocr\cap.png"
        pag.screenshot(path, region=(self.x1, self.y1, capture_width, capture_height))
        print("save")
        self.captureWindow.hide()


def captureWindow():
    global MainWindow
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Screen()
    sys.exit(app.exec_())

if __name__ == "__main__":
    MainWindow=None
    captureWindow()
    #https://www.python2.net/questions-266403.htm