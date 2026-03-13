from ultralytics import YOLO

model = YOLO("yolov8n.pt")
metrics_before = model.val(data="dataset.yaml", split="test")
print(metrics_before)