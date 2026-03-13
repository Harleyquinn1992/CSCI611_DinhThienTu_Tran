from ultralytics import YOLO
import cv2

MODEL_PATH = "runs/detect/train5/weights/best.pt"
VIDEO_PATH = "" #path to your video here

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: could not open video.")
    exit()

cv2.namedWindow("YOLO Video Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO Video Detection", 960, 540)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, imgsz=1280, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Video Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()