import os
import time

import cv2
from ultralytics import YOLO


VIDEO_PATH = os.path.join("videos", "traffic.mp4")
OUTPUT_PATH = os.path.join("output", "vehicle_counting_result.mp4")

CONFIDENCE_THRESHOLD = 0.4
LINE_Y = 340

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

TRACKED_CLASS_IDS = list(VEHICLE_CLASSES.keys())


def get_project_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )


def load_model():
    print("Loading YOLO model...")
    return YOLO("yolo11n.pt")

def open_video(video_path):

    cap = cv2.VideoCapture(video_path)

    # Start video processing from 49:00
    start_time_seconds = 49 * 60
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_seconds * 1000)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    return cap


def create_video_writer(output_path, width, height, fps):
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps if fps > 0 else 30.0,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open output video writer: {output_path}"
        )

    return writer


def calculate_center(x1, y1, x2, y2):
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    return center_x, center_y


def check_line_crossing(
    previous_y,
    current_y,
    track_id,
    counted_ids
):
    if previous_y is None:
        return False

    if track_id in counted_ids:
        return False

    # Vehicle moving from top to bottom
    crossed_down = (
        previous_y < LINE_Y
        and current_y >= LINE_Y
    )

    # Vehicle moving from bottom to top
    crossed_up = (
        previous_y > LINE_Y
        and current_y <= LINE_Y
    )

    return crossed_down or crossed_up
    return (
        previous_y is not None
        and previous_y < LINE_Y
        and current_y >= LINE_Y
        and track_id not in counted_ids
    )


def update_counts(
    track_id,
    class_id,
    vehicle_counts,
    counted_ids
):
    vehicle_type = VEHICLE_CLASSES.get(
        class_id,
        str(class_id)
    ).lower()

    if vehicle_type not in vehicle_counts:
        return sum(vehicle_counts.values())

    vehicle_counts[vehicle_type] += 1
    counted_ids.add(track_id)

    return sum(vehicle_counts.values())


def draw_statistics(
    frame,
    vehicle_counts,
    total_count,
    fps
):
    counts_text = (
        f"Cars: {vehicle_counts['car']}   "
        f"Motorcycles: {vehicle_counts['motorcycle']}   "
        f"Buses: {vehicle_counts['bus']}   "
        f"Trucks: {vehicle_counts['truck']}   "
        f"Total: {total_count}"
    )

    cv2.putText(
        frame,
        counts_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return frame


def process_frame(
    frame,
    model,
    previous_positions,
    counted_ids,
    vehicle_counts
):
    results = model.track(
        frame,
        persist=True,
        classes=TRACKED_CLASS_IDS,
        conf=CONFIDENCE_THRESHOLD,
        tracker="bytetrack.yaml",
        verbose=False,
        device=0
    )

    total_count = sum(vehicle_counts.values())

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return frame, total_count

    for box in result.boxes:

        cls_id = (
            int(box.cls[0])
            if box.cls is not None and len(box.cls) > 0
            else None
        )

        conf = (
            float(box.conf[0])
            if box.conf is not None and len(box.conf) > 0
            else 0.0
        )

        if cls_id is None:
            continue

        track_id = None

        if box.id is not None and len(box.id) > 0:
            track_id = int(box.id[0])
        else:
            print(
                "Warning: missing tracking ID for a detection."
            )

        if box.xyxy is None or len(box.xyxy) == 0:
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        center_x, center_y = calculate_center(
            x1,
            y1,
            x2,
            y2
        )

        if track_id is not None:

            previous_y = previous_positions.get(
                track_id
            )

            if check_line_crossing(
                previous_y,
                center_y,
                track_id,
                counted_ids
            ):
                total_count = update_counts(
                    track_id,
                    cls_id,
                    vehicle_counts,
                    counted_ids
                )

                print(
                    f"Vehicle ID {track_id} "
                    f"crossed the counting line."
                )

            previous_positions[track_id] = center_y

        label_name = VEHICLE_CLASSES.get(
            cls_id,
            str(cls_id)
        )

        label = (
            f"{label_name} "
            f"ID: {track_id if track_id is not None else 'N/A'} "
            f"| {conf:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.circle(
            frame,
            (int(center_x), int(center_y)),
            4,
            (0, 0, 255),
            -1
        )

    return frame, total_count


def main():

    project_root = get_project_root()

    video_path = os.path.abspath(
        os.path.join(
            project_root,
            VIDEO_PATH
        )
    )

    output_path = os.path.abspath(
        os.path.join(
            project_root,
            OUTPUT_PATH
        )
    )

    cap = None
    writer = None

    try:

        if not os.path.exists(video_path):
            raise FileNotFoundError(
                f"Input video does not exist: {video_path}"
            )

        print(f"Input video: {video_path}")

        cap = open_video(video_path)

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        input_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        output_fps = (
            input_fps
            if input_fps and input_fps > 0
            else 30.0
        )

        print(f"Video size: {width} x {height}")
        print(f"Video FPS: {output_fps}")

        writer = create_video_writer(
            output_path,
            width,
            height,
            output_fps
        )

        vehicle_counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }

        counted_ids = set()
        previous_positions = {}

        total_count = 0

        prev_frame_time = time.time()
        fps = 0.0

        frame_number = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                print("End of video reached.")
                break

            frame_number += 1

            frame, total_count = process_frame(
                frame,
                model,
                previous_positions,
                counted_ids,
                vehicle_counts
            )

            current_time = time.time()

            elapsed_time = (
                current_time -
                prev_frame_time
            )

            if elapsed_time > 0:

                current_fps = 1.0 / elapsed_time

                fps = (
                    fps * 0.9 +
                    current_fps * 0.1
                )

            prev_frame_time = current_time

            # Draw counting line
            cv2.line(
                frame,
                (0, LINE_Y),
                (frame.shape[1], LINE_Y),
                (0, 0, 255),
                2
            )

            # Draw statistics
            frame = draw_statistics(
                frame,
                vehicle_counts,
                total_count,
                fps
            )

            # Show video
            cv2.imshow(
                "Vehicle Counting",
                frame
            )

            # Save output frame
            writer.write(frame)

            # Press Q to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):

                print("User pressed Q. Exiting.")
                break

            # Progress
            if frame_number % 100 == 0:
                print(
                    f"Processed frames: {frame_number}"
                )

    except Exception as exc:

        print(f"Error: {exc}")

    finally:

        if writer is not None:
            writer.release()

        if cap is not None:
            cap.release()

        cv2.destroyAllWindows()

        print(
            f"Processed video saved to: {output_path}"
        )


if __name__ == "__main__":

    model = load_model()
    main()