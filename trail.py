from ultralytics import YOLO

model = YOLO("hemletYoloV8.pt")

print(model.names)
