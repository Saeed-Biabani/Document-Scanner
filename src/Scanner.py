import cv2
from numpy import ones, uint8
import numpy as np


class Scanner:
    def __init__(self):
        pass


    def Scan_Original(self, frame):
        self.Save_Scanned_Document(frame)


    def Scan_Black_Wite(self, frame):
        Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Filter = cv2.bilateralFilter(Gray, 25, 50, 100)
        TH = cv2.adaptiveThreshold(Filter, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)
        DIL = cv2.dilate(TH, kernel = ones((2, 2), uint8), iterations = 1)
        ERO = cv2.erode(DIL, kernel = ones((2, 2), uint8), iterations = 1)

        self.Save_Scanned_Document(ERO)


    def Scan_Enhaced(self, frame):
        Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Filter = cv2.bilateralFilter(Gray, 25, 50, 100)
        TH = cv2.adaptiveThreshold(Filter, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)
        DIL = cv2.dilate(TH, kernel = ones((2, 2), uint8), iterations = 1)
        ERO = cv2.erode(DIL, kernel = ones((2, 2), uint8), iterations = 1)

        mask = cv2.merge((ERO, ERO, ERO))
        Enhaced = cv2.bitwise_or(frame, mask)

        self.Save_Scanned_Document(Enhaced)
    
    
    def Save_Scanned_Document(self, Document):
        cv2.imwrite("result.jpg", Document)