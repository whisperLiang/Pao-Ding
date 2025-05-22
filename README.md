# Pao-Ding
Pao-Ding: Accelerating Cross-Edge Video Analytics via Automated CNN Model Partitioning.

This code is open for Automatic CNN Parsing module of Pao-Ding.

We will make all of our code publicly available to the community after publication.
## Getting Started
My python version: python3.10.12.
Install environment package.
```bash
pip3 install -r requirements.txt
```
## Automatic CNN Parsing for torchvision
```bash
python3 torchvision_split.py
```
## Automatic CNN Parsing for YOLOv5
Download YOLOv5 code.
```bash
git clone https://github.com/ultralytics/yolov5.git
```
Make sure that you can run detect.py correctly.
```bash
python3 detect.py
```
Merge all files and folders from Pao-Ding to yolov5.

Move detect_after_split.py from yolov5_split to yolov5, and then run it.
```bash
python3 detect_after_split.py
```
## Automatic CNN Parsing for YOLOv7
Download YOLOv7 code.
```bash
git clone https://github.com/WongKinYiu/yolov7.git
```
Make sure that you can run detect.py correctly.
```bash
python3 detect.py
```
Merge all files and folders from Pao-Ding to yolov7.

Move yolov7_detect_split.py from yolov7_split to yolov7, and then run it.
```bash
python3 yolov7_detect_split.py
```
