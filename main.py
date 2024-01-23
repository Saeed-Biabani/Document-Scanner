from Structure.getConfig import config_
from DocScanner import Scanner
from Utils import *
import pathlib
import os

scanner = Scanner("Structure/Scanner-Detector.pth", config_)

fname = pathlib.Path("image_file_name")

paper, org = ScannSavedImage(str(fname), scanner, True)

paper = EnhancePaper(paper)

save_path = pathlib.Path(os.path.join(fname.parent, "Resaults")); save_path.mkdir(exist_ok = True)

fname = os.path.join(str(save_path), fname.stem+"_det.jpg")
SaveCompImage(fname, org, paper)