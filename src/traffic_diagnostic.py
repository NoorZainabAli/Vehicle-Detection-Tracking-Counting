import cv2
from ultralytics import YOLO


VIDEO_PATH = "videos/traffic.mp4"

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


def main():

    print("Tracking movement diagnostic started")

    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Could not open video.")
        return

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(f"Total frames: {total_frames}")

    first_positions = {}
    last_positions = {}

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model.track(
            frame,
            persist=True,
            classes=list(VEHICLE_CLASSES.keys()),
            conf=0.2,
            tracker="bytetrack.yaml",
            verbose=False,
            device=0
        )[0]

        if (
            results.boxes is None
            or len(results.boxes) == 0
        ):
            continue

        for box in results.boxes:

            if box.id is None:
                continue

            track_id = int(box.id[0])

            cls_id = int(box.cls[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            position = (
                center_x,
                center_y
            )

            if track_id not in first_positions:

                first_positions[track_id] = {
                    "frame": frame_number,
                    "position": position,
                    "class": VEHICLE_CLASSES.get(
                        cls_id,
                        str(cls_id)
                    )
                }

            last_positions[track_id] = {
                "frame": frame_number,
                "position": position,
                "class": VEHICLE_CLASSES.get(
                    cls_id,
                    str(cls_id)
                )
            }

        if frame_number % 500 == 0:

            print(
                f"Processed frames: "
                f"{frame_number}/{total_frames}"
            )

    cap.release()

    print()
    print("=" * 70)
    print("VEHICLE MOVEMENT RESULTS")
    print("=" * 70)

    for track_id in first_positions:

        if track_id not in last_positions:
            continue

        first = first_positions[track_id]
        last = last_positions[track_id]

        x1, y1 = first["position"]
        x2, y2 = last["position"]

        dx = x2 - x1
        dy = y2 - y1

        print()
        print(f"Vehicle ID: {track_id}")
        print(f"Type: {first['class']}")

        print(
            f"First position: "
            f"({x1:.0f}, {y1:.0f})"
        )

        print(
            f"Last position: "
            f"({x2:.0f}, {y2:.0f})"
        )

        print(
            f"Movement: "
            f"dx={dx:.0f}, dy={dy:.0f}"
        )

        if abs(dx) > abs(dy):

            if dx > 0:
                direction = "LEFT -> RIGHT"
            else:
                direction = "RIGHT -> LEFT"

        else:

            if dy > 0:
                direction = "TOP -> BOTTOM"
            else:
                direction = "BOTTOM -> TOP"

        print(f"Direction: {direction}")

    print()
    print("Diagnostic completed.")


if __name__ == "__main__":
    main()