
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QCheckBox, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from Process import Process as Process_On_Document
from Scanner import Scanner as Scan_Document
from PyQt5.uic import loadUi
import sys
import cv2


class main(QMainWindow):
    def __init__(self):
        super(main , self).__init__()
        loadUi("GUI.ui" , self)

        self.open_file : QPushButton = self.findChild(QPushButton , "open_file")
        self.scan : QPushButton = self.findChild(QPushButton , "capture")
        self.scan_type : QComboBox = self.findChild(QComboBox , "scan_type")
        self.display_image : QLabel = self.findChild(QLabel , "display_image")

        self._Processor = Process_On_Document()
        self._Scanner = Scan_Document()

        self.open_file.clicked.connect(self.Take_Input_File)
        self.scan.clicked.connect(self.Preparation_For_Scan)



    def Take_Input_File(self):
        fname = QFileDialog.getOpenFileName(self, "Browse")

        try:
            if fname[0].split('.')[1] == "jpg" or fname[0].split('.')[1] == "png":
                self.OriginalImage = self._Processor.ReadImage(fname[0])
                self.scan.setEnabled(True)
                self.Detect_Document_AND_Draw_RECT()
        except:
            pass

    def Detect_Document_AND_Draw_RECT(self):
        Contours = self._Processor.Detect_Contours(self.OriginalImage)
        self.Biggest, Area = self._Processor.BiggestContoure(Contours)
        self.Drawed = self._Processor.DrawContour(self.Biggest, self.OriginalImage.copy())

        self.Display_On_Window(self.Drawed)

    def Preparation_For_Scan(self):
        Biggest = self._Processor.ReOrder(self.Biggest)
        Width, Height = self._Processor.Get_Height_Width(Biggest)
        self.WarpedFrame = self._Processor.GetWarpIMG(Biggest, self.OriginalImage, Height, Width)

        self.Scan_Document()

    def Scan_Document(self):
        Scan_Type = self.scan_type.currentText()
        # Types : Original, B&-W, Enhaced
        if Scan_Type == 'Original':
            self._Scanner.Scan_Original(self.WarpedFrame)

        elif Scan_Type == 'B&-W':
            self._Scanner.Scan_Black_Wite(self.WarpedFrame)
            
        elif Scan_Type == 'Enhaced':
            self._Scanner.Scan_Enhaced(self.WarpedFrame)


    def Display_On_Window(self, IMAGE = None):
        if IMAGE is None:
            IMAGE = self.Image
        
        image = QImage(IMAGE.data, IMAGE.shape[1], IMAGE.shape[0], QImage.Format_RGB888).rgbSwapped()
        self.display_image.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = main()
    window.show()
    sys.exit(app.exec_())
