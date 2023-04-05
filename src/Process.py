import cv2
from numpy import float32, ones, uint8, int32, array, zeros, argmin, argmax, diff
from math import sqrt

class Process:
    def __init__(self) -> None:
        self.Kernel = ones((5, 5), uint8)


    def ReadImage(self, filename):
        OriginalImage = cv2.imread(filename)
        return OriginalImage


    def ToGrayScale(self, Image):
        return cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)


    def Detect_Contours(self, Image):
        GrayImage = self.ToGrayScale(Image)

        Filter = cv2.bilateralFilter(GrayImage, 15, 100, 150)
        
        Edge = cv2.adaptiveThreshold(Filter, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 9)
        
        Dilate = cv2.dilate(Edge, self.Kernel, iterations = 2)
        Erode = cv2.erode(Dilate, self.Kernel, iterations = 1)
        
        Contours, Hierarchy = cv2.findContours(Erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        return Contours
    
    def BiggestContoure(self, Contours):
        Biggest = array([])
        maxArea = 0

        for cnt in Contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                Peri = cv2.arcLength(cnt, True)
                Approx = cv2.approxPolyDP(cnt, 0.02 * Peri, True)
                if area > maxArea and len(Approx) == 4:
                    Biggest = Approx
                    maxArea = area
        return Biggest, maxArea
    

    def DrawContour(self, CNT, frame):
        RES = frame

        if len(CNT) == 4:
            tmp = frame.copy()
            drawed = cv2.drawContours(tmp, [CNT], -1, (0, 255, 255), 25)
            FilledFrame = cv2.fillPoly(drawed, [CNT], (50, 150, 150))
            RES = cv2.addWeighted(frame, 0.6, FilledFrame, 0.4, 0)
        
        return RES
    

    def Get_Height_Width(self, Biggest):
        H = W = 0

        if len(Biggest) == 4:
            ptx_1, pty_1 = Biggest[0, 0]
            ptx_2, pty_2 = Biggest[1, 0]
            ptx_3, pty_3 = Biggest[2, 0]
            ptx_4, pty_4 = Biggest[3, 0]

            BW = sqrt((ptx_2 - ptx_4) ** 2 + (pty_2 - pty_4) ** 2)
            TW = sqrt((ptx_1 - ptx_3) ** 2 + (pty_1 - pty_3) ** 2)
            RH = sqrt((ptx_1 - ptx_2) ** 2 + (pty_1 - pty_2) ** 2)
            LH = sqrt((ptx_4 - ptx_3) ** 2 + (pty_4 - pty_3) ** 2)

            W = int(max(BW, TW))
            H = int(max(RH, LH))

        return H, W
    

    def ReOrder(self, myPoints):
        myPointsNew = myPoints

        if len(myPoints) == 4:
            myPoints = myPoints.reshape((4, 2))
            myPointsNew = zeros((4, 1, 2), int32)
            add = myPoints.sum(1)

            myPointsNew[0] = myPoints[argmin(add)]
            myPointsNew[3] = myPoints[argmax(add)]
            Diff = diff(myPoints, axis=1)
            myPointsNew[1] = myPoints[argmin(Diff)]
            myPointsNew[2] = myPoints[argmax(Diff)]

        return myPointsNew
    

    def GetWarpIMG(self, Biggest, Image, fH = 900, fW = 680):
        RES = Image

        if len(Biggest) == 4:
            pt1 = float32(Biggest)
            pt2 = float32([[0, 0], [fW, 0], [0, fH], [fW, fH]])
            Matrix = cv2.getPerspectiveTransform(pt1, pt2)
            RES = cv2.warpPerspective(Image, Matrix, (fW, fH))
        try:
            return RES
        except UnboundLocalError:
            raise Exception ("Can't Find Doncument! Try Another Image.")