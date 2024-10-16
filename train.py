from ultralytics import YOLO

import torch

if torch.cuda.is_available():
  print("PyTorch is using a GPU")
else:
  print("PyTorch is not using a GPU")

model = YOLO("yolov8n.yaml")

if __name__ == '__main__':
    results = model.train(data="config.yaml", epochs=5000, device='0')