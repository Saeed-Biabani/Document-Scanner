# Document-Scanner :bookmark_tabs:
- Fun And Useful Scanner. Based on OpenCV
## Installing dependencies:

- Execute `$ pip install opencv-python`
- Execute `$ pip install numpy`
- Execute `$ pip install math`

## Description:

![](/TestCases/testIMG3.jpg)
- Input Image

![](/TestCases/img.jpg)
- When The Yellow Line Drawed Around The Object It Means System Has Recognized it.
Now You Can Press 's' (Press Several Times) Button to Save The Scanned Document.

![](/TestCases/Result.jpg)
- [X] Result(BW, ORIGINAL, ENHACED)

## Run Python file:
- Flags : --Input-Type [IMAGE, RECORDED, LIVE], --Input, --Scan-Type [ORIGINAL, BW, ENHACED]

- [X] Use Image File : Execute `$ python3 main.py --Input-Type IMAGE --Input PathToFile/FileName.jpg --Scan-Type []`
- [X] Use Recordec Video File : Execute `$ python3 main.py --Input-Type RECORDED --Input PathToFile/FileName.mp4 --Scan-Type []`
- [X] Use Live Cam : Execute `$ python3 main.py --Input-Type LIVE --Scan-Type []`

> Press 'q' to kill all windows (if it's not work 'cntrl+c' in terminal :joy:)