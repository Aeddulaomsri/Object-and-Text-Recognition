Shop Sign Detection and Text Recognition in Video
This repository contains Python code for detecting shop signs and logos in video footage, followed by text recognition using Optical Character Recognition (OCR).

Functionality
Object Detection: The code utilizes a pre-trained YOLOv5 model to identify shop signs and logos within video frames.
Text Recognition: Extracted shop sign regions are processed using Tesseract OCR to recognize the shop names.
Data Compilation: Recognized shop names and corresponding timestamps are compiled into an Excel spreadsheet for further analysis.

Requirements
Python 3.x
OpenCV
PyTorch
Tesseract-OCR (with appropriate language packs if needed)
pandas
Pillow (PIL Fork)

How to Use:

Download a pre-trained YOLOv5 model:

You can download a YOLOv5 model from the Ultralytics repository: https://github.com/ultralytics/yolov5
Place the downloaded model files in the same directory as your script.
Set Tesseract path (if needed):

If Tesseract is not installed in a standard location, update the tesseract_cmd path in the code to point to the location of your Tesseract executable.
Run the script:

Modify the video_path variable in the main function to point to your video file.
Run the script using python script_name.py.
Output:

The script will process the video and generate an Excel spreadsheet named shop_data.xlsx containing recognized shop names and timestamps.
