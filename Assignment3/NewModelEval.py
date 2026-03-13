from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")

metrics_test = model.val(
    data="dataset.yaml",
    split="test",
    imgsz=1280,
    batch=1,
    device=0,
    workers=0
)

print(metrics_test.results_dict)