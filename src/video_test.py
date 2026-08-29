import os
import sys
import cv2


def main():
    # Go to the project root so we can find the video file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Try common video locations
    video_paths = [
        os.path.join(project_root, "videos", "traffic.mp4"),
        os.path.join(project_root, "data", "videos", "traffic.mp4"),
    ]

    video_path = None
    for path in video_paths:
        if os.path.exists(path):
            video_path = path
            break

    if video_path is None:
        print("Error: traffic.mp4 was not found.")
        print("Please place the video in either 'videos/traffic.mp4' or 'data/videos/traffic.mp4'.")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        sys.exit(1)

    print(f"Video opened successfully: {video_path}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("Traffic Video Test", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
