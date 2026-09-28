\# Vehicle Detection, Tracking \& Counting



A computer vision system for detecting, tracking, and counting vehicles in traffic videos using \*\*YOLO11\*\*, \*\*OpenCV\*\*, and \*\*ByteTrack\*\*.



The system processes traffic video frames and identifies four vehicle categories:



\* Car

\* Motorcycle

\* Bus

\* Truck



It provides bounding boxes, confidence scores, persistent tracking IDs, a virtual counting line, vehicle-wise counts, total vehicle count, FPS information, and a processed output video.



\## Features



\### 1. Vehicle Detection



YOLO11 is used to detect vehicles in each video frame.



The system currently detects:



| Class ID | Vehicle    |

| -------: | ---------- |

|        2 | Car        |

|        3 | Motorcycle |

|        5 | Bus        |

|        7 | Truck      |



A confidence threshold of \*\*0.4\*\* is used for vehicle detection.



Each detected vehicle is displayed with:



\* Bounding box

\* Vehicle class

\* Confidence score



\### 2. Vehicle Tracking



The project uses \*\*ByteTrack\*\* through Ultralytics tracking.



Each detected vehicle receives a tracking ID that is maintained across consecutive frames.



Example:



```text

Car ID: 12 | 0.91

Bus ID: 7 | 0.87

Truck ID: 15 | 0.94

```



This allows the system to distinguish individual vehicles while they move through the video.



\### 3. Vehicle Counting



The counting module uses a horizontal virtual line.



When a tracked vehicle crosses the counting line, its tracking ID is added to the set of counted vehicles.



This prevents the same tracked vehicle from being counted repeatedly.



The system maintains separate counts for:



\* Cars

\* Motorcycles

\* Buses

\* Trucks

\* Total vehicles



\### 4. Real-Time Statistics



The processed video displays:



```text

Cars: 10   Motorcycles: 2   Buses: 3   Trucks: 5   Total: 20

FPS: 24.5

```



\### 5. Output Video



The counting module saves the processed video to:



```text

output/vehicle\_counting\_result.mp4

```



The output contains:



\* Vehicle bounding boxes

\* Vehicle labels

\* Tracking IDs

\* Confidence scores

\* Counting line

\* Vehicle statistics

\* FPS



\---



\## System Workflow



```text

Traffic Video

&#x20;     ↓

OpenCV Frame Extraction

&#x20;     ↓

YOLO11 Vehicle Detection

&#x20;     ↓

Vehicle Class Filtering

&#x20;     ↓

ByteTrack Object Tracking

&#x20;     ↓

Unique Vehicle IDs

&#x20;     ↓

Counting Line Crossing

&#x20;     ↓

Vehicle-wise Counting

&#x20;     ↓

Statistics + FPS

&#x20;     ↓

Processed Output Video

```



\## Technologies



\* \*\*Python\*\*

\* \*\*YOLO11\*\*

\* \*\*Ultralytics\*\*

\* \*\*OpenCV\*\*

\* \*\*ByteTrack\*\*

\* \*\*NumPy\*\*

\* \*\*PyTorch\*\*

\* \*\*Torchvision\*\*



\## Project Structure



```text

Vehicle\_Detection\_Counting/

│

├── src/

│   ├── traffic\_diagnostic.py

│   ├── vehicle\_counting.py

│   ├── vehicle\_tracking.py

│   ├── video\_detection.py

│   ├── video\_test.py

│   ├── yolo\_frame\_test.py

│   └── yolo\_import\_test.py

│

├── videos/

│   └── traffic.mp4

│

├── output/

│   └── vehicle\_counting\_result.mp4

│

├── .gitignore

├── README.md

└── requirements.txt

```



> Large video files, model weights, generated outputs, virtual environments, and temporary files are excluded from version control through `.gitignore`.



\## Installation



\### 1. Clone the repository



```bash

git clone <YOUR\_GITHUB\_REPOSITORY\_URL>

cd Vehicle\_Detection\_Counting

```



\### 2. Create a virtual environment



Windows:



```powershell

python -m venv .venv

```



Activate it:



```powershell

.venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```powershell

python -m pip install -r requirements.txt

```



The main dependencies are:



```text

ultralytics==8.4.71

opencv-python==4.13.0.92

numpy==2.2.6

torch==2.6.0

torchvision==0.21.0

```



\## GPU Support



The project supports NVIDIA GPU acceleration through PyTorch and Ultralytics.



For NVIDIA GPU users, verify CUDA availability with:



```powershell

python -c "import torch; print('CUDA available:', torch.cuda.is\_available()); print('GPU:', torch.cuda.get\_device\_name(0) if torch.cuda.is\_available() else 'CPU')"

```



Example:



```text

CUDA available: True

GPU: NVIDIA GeForce RTX 4050 Laptop GPU

```



The vehicle counting module explicitly uses:



```python

device=0

```



for YOLO tracking inference.



\## Input Video



Place your traffic video inside:



```text

videos/traffic.mp4

```



The project expects the following path:



```text

Vehicle\_Detection\_Counting/

└── videos/

&#x20;   └── traffic.mp4

```



Large video files should normally not be committed to GitHub. Add your own compatible traffic video locally.



\## Running the Project



\### Vehicle Detection



Run:



```powershell

python src\\video\_detection.py

```



This module performs YOLO11 detection and displays:



\* Vehicle bounding boxes

\* Vehicle class

\* Confidence score



Press:



```text

Q

```



to stop the video.



\### Vehicle Tracking



Run:



```powershell

python src\\vehicle\_tracking.py

```



This module uses ByteTrack to assign tracking IDs to vehicles.



Example:



```text

Car ID: 3 | 0.92

Truck ID: 8 | 0.89

Bus ID: 5 | 0.94

```



Press `Q` to exit.



\### Vehicle Counting



Run:



```powershell

python src\\vehicle\_counting.py

```



The counting module:



1\. Opens the traffic video.

2\. Starts processing from the configured video timestamp.

3\. Detects vehicles using YOLO11.

4\. Tracks vehicles using ByteTrack.

5\. Calculates vehicle center points.

6\. Detects counting-line crossings.

7\. Counts each tracked vehicle once.

8\. Displays vehicle statistics.

9\. Saves the processed video.



The current implementation starts video processing from:



```text

49:00

```



This is configured in `vehicle\_counting.py`.



The counting line is currently positioned at:



```python

LINE\_Y = 340

```



The output is saved as:



```text

output/vehicle\_counting\_result.mp4

```



\## Counting Logic



The system tracks the vertical center position of each vehicle.



For a vehicle with tracking ID `ID`, the previous and current Y-coordinates are compared with the counting line.



A vehicle is counted when it crosses the line either:



```text

Top → Bottom

```



or:



```text

Bottom → Top

```



Each tracking ID is stored in a set:



```python

counted\_ids

```



so the same vehicle is not counted multiple times.



\## Configuration



The main parameters can be modified in:



```text

src/vehicle\_counting.py

```



\### Confidence Threshold



```python

CONFIDENCE\_THRESHOLD = 0.4

```



\### Counting Line



```python

LINE\_Y = 340

```



\### Vehicle Classes



```python

VEHICLE\_CLASSES = {

&#x20;   2: "Car",

&#x20;   3: "Motorcycle",

&#x20;   5: "Bus",

&#x20;   7: "Truck",

}

```



\### Tracker



```python

tracker="bytetrack.yaml"

```



\## Testing



The repository also contains diagnostic and testing scripts:



```text

src/

├── traffic\_diagnostic.py

├── video\_test.py

├── yolo\_frame\_test.py

└── yolo\_import\_test.py

```



These scripts can be used to verify video loading, YOLO model loading, frame processing, and environment configuration.



\## Output



The vehicle counting pipeline generates a processed video containing annotated traffic frames.



Example annotation format:



```text

Car ID: 14 | 0.93

```



and the statistics overlay:



```text

Cars: 25   Motorcycles: 4   Buses: 6   Trucks: 8   Total: 43

FPS: 21.7

```



\## Limitations



The current implementation is designed as a practical vehicle detection, tracking, and counting pipeline and has several configurable assumptions:



\* The counting line has a fixed Y-coordinate.

\* The input video path is configured in the source code.

\* The counting module currently starts from a fixed timestamp.

\* Counting accuracy depends on detection and tracking quality.

\* Heavy traffic, occlusion, camera movement, and missed detections can affect tracking and counting.

\* The current implementation uses the pretrained YOLO11 model without project-specific vehicle fine-tuning.



\## Future Improvements



Potential extensions include:



\* Automatic counting-line placement

\* Multiple counting lines

\* Direction-wise traffic analysis

\* Vehicle speed estimation

\* Lane-wise vehicle counting

\* Traffic density estimation

\* Vehicle trajectory visualization

\* CSV/JSON export of vehicle statistics

\* Dashboard integration

\* Custom-trained YOLO model

\* Advanced tracking and re-identification

\* Real-time camera/RTSP stream support



\## Author



\*\*Zainab Ali\*\*



MS Data Science | Computer Vision | Machine Learning | Artificial Intelligence



\## License



This project is intended for educational and research purposes.



