import cv2
import numpy as np
from PIL import ImageGrab
from ultralytics import YOLO

import datetime
import time
import os

# define some constants
CONFIDENCE_THRESHOLD = 0.8
SCREEN_WIGTH = 1920
SCREEN_HEIGHT = 1080
GREEN = (0, 255, 0)

FILE_TO_SAVE_IMG = "D:/Projects/CS-GOObjectDetection/datasets/tmp/ToSortProcessed/"

def CreateYOLOdata(class_id, xmin, ymin, xmax, ymax, file_name):
    width = (xmax - xmin)
    height = (ymax - ymin)
    xcentre = xmax - width / 2
    ycentre = ymax - height / 2

    xcentre = round(xcentre / SCREEN_WIGTH, 6)
    width = round(width / SCREEN_WIGTH, 6)

    ycentre = round(ycentre / SCREEN_HEIGHT, 6)
    height = round(height / SCREEN_HEIGHT, 6)

    with open(file_name + ".txt", 'a') as file:
        file.write(f"{str(int(class_id))} {str(float(xcentre))} {str(float(ycentre))} {str(float(width))} {str(float(height))} \n")

model = YOLO("D:/Projects/CS-GOObjectDetection/runs/detect/train25/weights/best.pt")

while True:
    start = datetime.datetime.now()

    # Capture the screen as a numpy array
    screen = np.array(ImageGrab.grab(bbox=(0, 0, 1920 , 1080 ))) # Adjust the bounding box as needed

    # Convert the BGR image to RGB
    frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    # run the YOLO model on the frame
    detections = model(frame)[0]

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # extract the confidence (i.e., probability) associated with the detection
        confidence = data[4]

        # filter out weak detections by ensuring the 
        # confidence is greater than the minimum confidence
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        # if the confidence is greater than the minimum confidence,
        # draw the bounding box on the frame
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)

        # Save image
        file_path = FILE_TO_SAVE_IMG + timestamp

        if not os.path.exists(file_path + ".jpg"):
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            cv2.imwrite(file_path + ".jpg", screen)

        # Save coordinates for the image
        CreateYOLOdata(data[5], xmin, ymin, xmax, ymax, file_path)

    # end time to compute the fps
    end = datetime.datetime.now()
    # show the time it took to process 1 frame
    total = (end - start).total_seconds()
    print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")

    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / total:.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

    # Display the captured screen
    cv2.imshow('Screen Capture', frame)

    time.sleep(1)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the OpenCV window and close it
cv2.destroyAllWindows()