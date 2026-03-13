from ultralytics import YOLO

# pretrained model
model = YOLO("yolov8n.pt")

# Train model
model.train(
    data="dataset.yaml",
    epochs=30,
    imgsz=1280,
    batch=2,
    device=0,
    workers=0
)

# Load best trained weights automatically
fine_tuned = YOLO(model.trainer.best)

# Evaluate on test dataset
metrics_after = fine_tuned.val(data="dataset.yaml", split="test")

print("Precision:", metrics_after.results_dict["metrics/precision(B)"])
print("Recall:", metrics_after.results_dict["metrics/recall(B)"])
print("mAP50:", metrics_after.results_dict["metrics/mAP50(B)"])
print("mAP50-95:", metrics_after.results_dict["metrics/mAP50-95(B)"])