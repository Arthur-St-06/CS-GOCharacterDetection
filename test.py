from ultralytics import YOLO

import torch

model = YOLO("D:/Projects/CS-GOObjectDetection/runs/detect/train17/weights/best.pt")

if __name__ == '__main__':
    while True:
        results = model.predict(source="D:/Projects/CS-GOObjectDetection/datasets/coco/images/validate", show=True)

    #results = model.train(data="config.yaml", epochs=500, device='0')