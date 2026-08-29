# Vehicle Detection and Counting

## 1. Project Overview

This project is a beginner-friendly computer vision system for traffic analysis. It reads a video, detects vehicles, tracks each one across frames, and counts vehicles when they cross a virtual line.

The system uses YOLO11 for object detection and ByteTrack for object tracking. This allows the program to recognize a vehicle in one frame and keep the same identity in later frames. A horizontal virtual line is drawn in the video, and the system counts a vehicle when it moves from above the line to below the line.

This project is a good introduction to real-world computer vision workflows, especially for traffic monitoring, traffic flow analysis, and smart city applications.

## 2. Features

- Real-time vehicle detection
- Car detection
- Motorcycle detection
- Bus detection
- Truck detection
- Object tracking
- Unique tracking IDs
- Virtual counting line
- Line-crossing detection
- Duplicate-count prevention
- Per-class vehicle counting
- Total vehicle counting
- FPS monitoring
- Processed output video

## 3. Technology Stack

| Technology | Purpose |
| --- | --- |
| Python | Main programming language |
| OpenCV | Video processing and visualization |
| YOLO11 | Vehicle detection |
| ByteTrack | Object tracking |
| NumPy | Numerical operations |

## 4. How the System Works

The system follows this pipeline:

Traffic Video
→ OpenCV
→ YOLO11
→ Vehicle Detection
→ ByteTrack
→ Tracking IDs
→ Center Point
→ Counting Line
→ Line Crossing
→ Vehicle Count
→ Output Video

Each stage is important:

1. OpenCV reads the video frame-by-frame.
2. YOLO11 looks at each frame and detects vehicles.
3. ByteTrack keeps the same vehicle identity across multiple frames.
4. The system calculates the center of each bounding box.
5. A horizontal line is used as a counting boundary.
6. When a tracked vehicle moves from above the line to below the line, it is counted once.
7. The program displays the results on the video and saves the processed output.

## 5. Vehicle Classes

| Class ID | Vehicle |
| --- | --- |
| 2 | Car |
| 3 | Motorcycle |
| 5 | Bus |
| 7 | Truck |

These class IDs correspond to the standard COCO-style object classes used by YOLO.

## 6. Counting Logic

The counting logic is based on the center point of each tracked vehicle and a fixed virtual line.

1. YOLO detects a vehicle.
2. ByteTrack assigns a tracking ID to that object.
3. The system calculates the center point of the bounding box.
4. The previous center position is stored for that tracking ID.
5. The current center position is compared with `LINE_Y`.
6. If the vehicle moves from above the line to below the line, it is treated as a crossing.
7. `counted_ids` prevents the same tracking ID from being counted more than once.
8. The correct vehicle-type counter is increased.

The main crossing condition is:

`previous_y < LINE_Y and current_y >= LINE_Y`

Example:

- A vehicle had a center point at `y = 300` in the previous frame.
- In the current frame, its center point is at `y = 370`.
- Since `300 < 350` and `370 >= 350`, the vehicle has crossed the line.
- The vehicle is counted once and its tracking ID is stored in `counted_ids`.

This prevents the same vehicle from being counted repeatedly while it remains below the line in later frames.

## 7. Project Structure

```text
Vehicle_Detection_Counting/
│
├── videos/
│   └── traffic.mp4
│
├── src/
│   ├── video_test.py
│   ├── yolo_frame_test.py
│   ├── video_detection.py
│   ├── vehicle_tracking.py
│   └── vehicle_counting.py
│
├── models/
├── output/
├── requirements.txt
├── README.md
└── .gitignore
```

### Important folders and files

- `videos/` stores the input traffic video used for testing and counting.
- `src/` contains the different Python scripts created step by step during the project.
- `video_test.py` checks whether the video can be opened and displayed correctly.
- `yolo_frame_test.py` tests YOLO on a single frame before full-video processing.
- `video_detection.py` runs YOLO detection on the complete video.
- `vehicle_tracking.py` adds tracking IDs with ByteTrack.
- `vehicle_counting.py` is the main project script that detects, tracks, and counts vehicles crossing a virtual line.
- `models/` is used for model files and weights if needed.
- `output/` stores the final processed video such as `vehicle_counting_result.mp4`.
- `requirements.txt` contains the Python dependencies for the project.
- `.gitignore` keeps unnecessary files out of version control.

## 8. Installation

Create and activate a virtual environment in Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Then install the project dependencies:

```powershell
pip install -r requirements.txt
```

## 9. Input Video

Place the traffic video in:

```text
videos/traffic.mp4
```

The program expects this path to exist before running.

## 10. Running the Project

From the project root, run:

```powershell
python src/vehicle_counting.py
```

The program processes the video, draws the results on the frames, and saves the final processed video to the output folder.

## 11. Output

The processed result video is saved to:

```text
output/vehicle_counting_result.mp4
```

The output video contains:

- bounding boxes
- vehicle labels
- tracking IDs
- center points
- counting line
- vehicle counts
- FPS

## 12. Configuration

The project uses a few key configuration values defined near the top of the main script:

- `VIDEO_PATH` - path to the input video
- `OUTPUT_PATH` - path for saving the processed output video
- `CONFIDENCE_THRESHOLD` - minimum YOLO confidence required for detection
- `LINE_Y` - the vertical position of the counting line
- `VEHICLE_CLASSES` - the set of classes used for vehicle detection and counting

Changing `LINE_Y` changes where the counting line appears in the frame. For example, moving it lower or higher can make the system count vehicles at different positions in the road.

## 13. Example Result

The following values are example values for demonstration only and are not measured benchmark results:

```text
Cars: 15
Motorcycles: 4
Buses: 2
Trucks: 4
Total: 25
FPS: 28.5
```

These values represent a sample output and help show how the statistics are displayed on the processed video.

## 14. Limitations

This system is useful for learning and basic traffic counting, but it also has realistic limitations:

- Heavy occlusion can cause tracking IDs to change or disappear temporarily.
- Poor lighting can reduce detection quality.
- Camera angle and camera motion can affect the counting accuracy.
- Crowded scenes can make tracking more difficult.
- The current system counts vehicles moving from above the line to below the line only.

## 15. Future Improvements

Possible improvements for later versions include:

- Two-direction counting
- Multiple counting lines
- Vehicle speed estimation
- RTSP/CCTV camera support
- Real-time dashboard
- Database integration
- Custom vehicle dataset
- Model fine-tuning
- Traffic analytics
- Vehicle heatmaps

## 16. Learning Outcomes

This project teaches core computer vision and Python concepts, including:

- Video processing
- OpenCV
- Object detection
- YOLO
- Object tracking
- ByteTrack
- Bounding boxes
- Confidence scores
- Tracking IDs
- Center-point calculation
- Line-crossing logic
- Data structures such as dictionaries and sets
- FPS measurement
- VideoWriter
- Python project organization

## 17. License

This project code is released under a simple open-source license for educational and portfolio use. The YOLO model and associated third-party libraries are subject to their own licenses and terms of use. This project does not claim ownership of those third-party model/software licenses.

If you want to use this project in a professional or commercial setting, check the licensing terms for the underlying libraries, tools, and model weights before deployment.
