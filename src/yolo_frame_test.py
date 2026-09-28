import cv2
from ultralytics import YOLO


def main():
    print("YOLO middle-frame test started")

    # Load YOLO11 nano model
    model = YOLO("yolo11n.pt")

    # Open traffic video
    cap = cv2.VideoCapture("videos/traffic.mp4")

    if not cap.isOpened():
        print("Error: Could not open videos/traffic.mp4")
        return

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total frames: {total_frames}")

    # Select middle frame
    middle_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

    # Read middle frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read middle frame")
        cap.release()
        return

    print(f"Testing frame: {middle_frame}")

    # Save middle frame
    cv2.imwrite("middle_frame.jpg", frame)
    print("Middle frame saved as middle_frame.jpg")

    # Vehicle classes from COCO
    vehicle_classes = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    # Detection confidence
    conf_threshold = 0.2

    # YOLO detection on RTX 4050 GPU
    results = model(
        frame,
        conf=conf_threshold,
        classes=list(vehicle_classes.keys()),
        verbose=False,
        device=0
    )[0]

    # Draw detections
    for box in results.boxes:

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        label = vehicle_classes.get(cls_id, str(cls_id))

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    print(f"Detected vehicles: {len(results.boxes)}")

    # Display result
    cv2.imshow("YOLO Middle Frame Test", frame)

    print("Press Q to close the window.")

    while True:
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()