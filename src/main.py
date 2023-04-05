import cv2
from Process import Process as Process_On_Document
from Scanner import Scanner as Scan_Document
from sys import argv
from argparse import ArgumentParser

_Processor = Process_On_Document()
_Scanner = Scan_Document()
PARSER = ArgumentParser()
CAM_STATUS = False


PARSER.add_argument("-IT", "--Input-Type", dest = "input_type")
PARSER.add_argument("-I", "--Input", dest = "input")
PARSER.add_argument("-S", "--Scan-Type", dest= "scan_type")

FLAG = PARSER.parse_args()


def Save_Result(Biggest):
    Biggest = _Processor.ReOrder(Biggest)
    Width, Height = _Processor.Get_Height_Width(Biggest)
    WarpedFrame = _Processor.GetWarpIMG(Biggest, frame, Height, Width)

    if FLAG.scan_type == "ORIGINAL":
        _Scanner.Scan_Original(WarpedFrame)

    elif FLAG.scan_type == "BW":
        _Scanner.Scan_Black_Wite(WarpedFrame)
        
    elif FLAG.scan_type == "ENHACED":
        _Scanner.Scan_Enhaced(WarpedFrame)


if FLAG.input_type == "LIVE":
    SName = "RES"
    cap = cv2.VideoCapture(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    CAM_STATUS, Orgframe = cap.read()

elif FLAG.input_type == "RECORDED":
    fname = FLAG.input
    SName = fname.split('.')[0]
    cap = cv2.VideoCapture(fname)

    CAM_STATUS, Orgframe = cap.read()

elif FLAG.input_type == "IMAGE":
    fname = FLAG.input
    SName = fname.split('.')[0]+"_RES"
    img = _Processor.ReadImage(fname)
    Orgframe = img.copy()


while True:
    if CAM_STATUS:
        CAM_STATUS, Orgframe = cap.read()
    frame = Orgframe.copy()

    Contours = _Processor.Detect_Contours(frame)
    Biggest, Area = _Processor.BiggestContoure(Contours)
    Drawed = _Processor.DrawContour(Biggest, frame.copy())

    cv2.imshow("Scan", Drawed)

    if cv2.waitKey(1) == ord('s'):
        Save_Result(Biggest)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()