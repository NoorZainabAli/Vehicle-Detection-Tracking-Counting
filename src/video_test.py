import os
import sys
import cv2
from ultralytics import YOLO


def main():

    # -----------------------------
    # Project paths
    # -----------------------------
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )

    video_path = os.path.join(
        project_root, "videos", "traffic.mp4"
    )

    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    # -----------------------------
    # Load YOLO model
    # -----------------------------
    model = YOLO("yolo11n.pt")

    # Vehicle classes from COCO
    vehicle_classes = [2, 3, 5, 7]

    # -----------------------------
    # Open video
    # -----------------------------
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    print(f"Video opened successfully: {video_path}")
    print("Starting YOLO vehicle detection...")
    print("Press Q to quit.")

    # -----------------------------
    # Process video
    # -----------------------------
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # YOLO inference on GPU
        results = model.predict(
            source=frame,
            conf=0.4,
            classes=vehicle_classes,
            device=0,
            verbose=False
        )

        # Draw bounding boxes
        annotated_frame = results[0].plot()

        cv2.imshow(
            "YOLO Vehicle Detection",
            annotated_frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # -----------------------------
    # Cleanup
    # -----------------------------
    cap.release()
    cv2.destroyAllWindows()

    print("Detection finished.")


if __name__ == "__main__":
    main()