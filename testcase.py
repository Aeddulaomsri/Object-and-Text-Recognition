import cv2
import torch
import pytesseract
import pandas as pd
from PIL import Image


# Tesseract OCR engine 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' #path to .exe file


# Loading Pre-trained YOlOV5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def detect_objects(frame):
    image = Image.fromarray(frame)  # Converting frame to PIL image

    results = model(image) # Performing object detection with model

    detections = results.pandas().xyxy[0]   # Applying Non-Maximal Suppression (NMS)
    detections = detections[detections['confidence'] > 0.7]  # Filter by confidence

    # Extracting bounding boxes and class labels
    bboxes = detections[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    labels = detections['name'].tolist()

    return bboxes, labels

def pre_process_image(image):
   
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converting to grayscale

  gray = cv2.filter2D(gray,3,5,5)  #Filtering the noise, chnage accoringdly 
   
  return gray


#  Text recognition using OCR
def recognize_text(image):
    text = pytesseract.image_to_string(image) # Applying OCR to the input image
    return text.strip()

# Processing video frames
def process_video(video_path):
    video = cv2.VideoCapture(video_path) # Opening video file

    shop_data = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        boxes, labels = detect_objects(frame)  # Performing object detection to detect shop signs and logos

        # Extracting text from detected regions using OCR
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = [int(val) for val in box] 
            shop_image = pre_process_image(frame[int(y_min):int(y_max), int(x_min):int(x_max)])
            shop_name = recognize_text(shop_image)
            shop_data.append({'Shop Name': shop_name, 'Time Frame': video.get(cv2.CAP_PROP_POS_MSEC) / 1000})

            # Displaying bounding boxes  
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, str(labels[i]), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()    # Closing video file

    return shop_data

# Compiling into Excel spreadsheet
def compile_data(data, output_file):
    
    df = pd.DataFrame(data) # Converting data into DataFrame

    df.to_excel(output_file, index=False) # Writing to Excel spreadsheet

    print("Compilation completed. Output saved as", output_file)



def main():
    video_path = 'G:\Video1.mp4' # Input video file path

    shop_data = process_video(video_path) # Processing video frames

    # Compiling data into Excel spreadsheet
    output_file = 'shop_data1.xlsx'
    compile_data(shop_data, output_file)

if __name__ == "__main__":
    main()
