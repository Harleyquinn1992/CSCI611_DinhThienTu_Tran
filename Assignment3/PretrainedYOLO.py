from ultralytics import YOLO
import cv2


model = YOLO("yolov8n.pt") # Load pre-trained YOLO model

results = model("test.png", conf = 0.5)
annotated = results[0].plot()
cv2.imwrite("test_result.png", annotated)
