import cv2
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture("videos/traffic.mp4")

    if not cap.isOpened():
        print("Error: Could not open videos/traffic.mp4")
        return

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read the first frame from videos/traffic.mp4")
        cap.release()
        return

    vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    conf_threshold = 0.4

    results = model(frame, conf=conf_threshold, classes=list(vehicle_classes.keys()), verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        label = vehicle_classes.get(cls_id, str(cls_id))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imshow("YOLO Frame Test", frame)

    while True:
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
