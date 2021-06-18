import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtCore
from PIL.ImageQt import ImageQt 

from PIL import Image      

from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys

import pyautogui as pag

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QGridLayout, QWidget, QLabel, QMainWindow

import urllib.request
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

from inference import  LInference

import threading


form_class = uic.loadUiType("main.ui")[0]

class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__(None, QtCore.Qt.WindowStaysOnTopHint)
        self.setupUi(self)
        
        self.x1 = 0
        self.y1 = 0
        
        self.x2 = 0
        self.y2 = 0
       
        self.resize(780, 500)
        self.mode = 0
        self.draw =0
        #self.label_capture.setPixmap(QPixmap("screen.png"))
        #self.label.setScaledContents(True)
        
        self.pushButton.clicked.connect(self.openWindown)

        self.latext.setText("")
        self.show()
    
  
    def showMainWidgets(self):
        print("showMainWidgets")
        self.centralwidget.show()

    def hideMainWidgets(self):
        print("hideMainWidgets")
        self.centralwidget.hide()

    def openWindown(self):
        self.hideMainWidgets()
        self.setCursor(QCursor(Qt.CrossCursor))
        self.setWindowOpacity(0.2)
        self.showMaximized()
        self.mode =1
        

        
    def capture(self):
        capture_width = self.x2 - self.x1
        capture_height = self.y2 - self.y1
        
        if capture_width  < 5 or capture_height < 5 :
            return

        self.mode = 0 
        print(self.mode)
    
        image=pag.screenshot(region=(self.x1, self.y1+28, capture_width, capture_height))

        
        print("save")
        

        self.showMainWidgets()


        imageq = ImageQt(image)
        pixmap = QtGui.QPixmap.fromImage(imageq)
        pixmap = pixmap.scaled(QSize(800, 400), QtCore.Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.label_capture.setPixmap(pixmap)
        self.setCursor(QCursor(Qt.ArrowCursor))
        
        self.showNormal()
        self.setWindowOpacity(1)

        data = ""
        data = inf.getLatext(image)[0]
        print(data)
        

        #self.label.setScaledContents(True)

        #desc ="x= \\frac {-b^{\prime} \pm \sqrt{b -ac}}{a}"

        self.latext.setText(data)
        # urlString = "http://www.sciweavers.org/tex2img.php?eq=%5Csqrt%5B3%5D%7Bx%5E3%2By%5E3%20%5Cover%202%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0"
        # url = urlparse(urlString)

        # qs = dict(parse_qsl(url.query))
        # qs['eq'] = data
        # parts = url._replace(query=urlencode(qs))
        # url = urlunparse(parts)
        
        # img = urllib.request.urlopen(url).read()

        # pixmap = QPixmap()
        # pixmap.loadFromData(img)

        # pixmap.scaled(QSize(400, 300), QtCore.Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # self.label_image.setPixmap(pixmap)
        print("end")


    def mouseMoveEvent(self, event):
        if self.mode ==1:
            self.x2, self.y2 = event.pos().x(), event.pos().y() 
            self.update()


    def paintEvent(self, event):
        
        width = self.x2-self.x1 + 6
        height = self.y2 - self.y1 +6

        if width < 1  or height  < 1:
            return    

        if self.draw == 1:
            qp = QPainter()
            qp.begin(self)
            qp.setPen(QPen(Qt.red, 4, Qt.SolidLine))           
            qp.drawRect(self.x1-5, self.y1-5, width, height)        
            qp.end()
        else:
            qp = QPainter()
            qp.begin(self)
            r = QtCore.QRect(self.x1-30,self.y2, width+200,height+100)
            qp.eraseRect(r)
            qp.end()
          

        
    def mousePressEvent(self, event):
        print("pressed")
        if event.button() == Qt.LeftButton:
            self.x1 = event.x()
            self.y1 = event.y()
            self.draw = 1

  

    def mouseReleaseEvent(self, event):
        print("release")
        self.draw = 0
        if event.button() == Qt.LeftButton:
            if self.mode == 1:
                self.x2 = event.x()
                self.y2 = event.y()
                self.update()
                self.capture()


def runWindow():
    app = QApplication(sys.argv) 
    myWindow = WindowClass() 
    myWindow.show()
    app.exec_()

if __name__ == "__main__" :
    inf = LInference()
    runWindow()
    