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
RED = (255, 0, 0)
GREEN = (0, 255, 0)

model = YOLO("D:/Projects/CS-GOObjectDetection/runs/detect/train22/weights/best.pt")

while True:
    start = datetime.datetime.now()

    # Capture the screen as a numpy array
    screen = np.array(ImageGrab.grab(bbox=(0, 0, 1920 , 1080 ))) # Adjust the bounding box as needed

    # Convert the BGR image to RGB
    frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    # run the YOLO model on the frame
    detections = model(frame)[0]

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
        if data[5] == 0 or data[5] == 1:
            cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)
        else:
            cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), RED, 2)

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

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the OpenCV window and close it
cv2.destroyAllWindows()