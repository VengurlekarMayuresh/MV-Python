# YOLO Object Detection with Distance Estimation - Complete Guide

## Table of Contents
1. [Project Overview](#project-overview)
2. [How It Works](#how-it-works)
3. [Prerequisites & Installation](#prerequisites--installation)
4. [Project Structure](#project-structure)
5. [Step-by-Step Setup](#step-by-step-setup)
6. [Code Explanation](#code-explanation)
7. [Customization](#customization)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project uses **YOLOv8 (You Only Look Once)** - a state-of-the-art real-time object detection system - to detect specific objects (person, cell phone, book) and estimate their distance from the camera.

### Key Features:
- Real-time object detection using webcam
- Distance estimation based on object size
- Audio alert when objects are too close (< 30 inches)
- Video recording capability
- Supports multiple target classes (person, cell phone, book)

---

## How It Works

### The Science Behind Distance Estimation

The system estimates distance using the **pinhole camera model** principle:

**Formula:** `Distance = (Real_Object_Width × Focal_Length) / Width_in_Frame`

Where:
- **Real_Object_Width**: Known physical width of the object (in inches)
- **Focal_Length**: Camera's focal length (calculated once using reference images)
- **Width_in_Frame**: Object's width in pixels (detected by YOLO)

### Process Flow:
```
1. Calibration Phase:
   - Take reference photos of person & phone at KNOWN_DISTANCE (45 inches)
   - System detects objects → measures pixel width
   - Calculates focal length: F = (pixel_width × 45) / real_width

2. Detection Phase (Real-time):
   - YOLO detects objects in video frame
   - System measures object pixel width
   - Distance = (real_width × focal_length) / pixel_width
   - Display distance & trigger alert if too close
```

---

## Prerequisites & Installation

### System Requirements:
- Python 3.8+
- Webcam
- 500MB free disk space

### Step 1: Clone/Download the Project

```bash
# Extract or clone the project folder
cd "MV Python"
```

### Step 2: Set Up Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the required packages:

```bash
pip install opencv-python
pip install ultralytics
pip install numpy
pip install pyttsx3  # For audio alerts (optional)
```

**Dependencies Explained:**
- `opencv-python` (cv2): Image processing, camera access, drawing
- `ultralytics`: YOLOv8 implementation
- `numpy`: Numerical operations
- `pyttsx3`: Text-to-speech for audio alerts

---

## Project Structure

```
MV Python/
├── classes.txt              # List of detectable classes (3 classes)
├── yolov8n.pt              # Pre-trained YOLOv8 nano model (auto-downloaded)
├── DistanceEstimation.py   # Main script with distance calculation
├── demo_Recoder.py         # Script to record reference videos/images
├── CaptureReferenceImage.py # Script for capturing reference images
├── ReferenceImages/        # Folder containing reference images
│   ├── image4.png         # Reference for cell phone
│   └── image14.png        # Reference for person
└── PROJECT_GUIDE.md       # This file
```

---

## Step-by-Step Setup

### STEP 1: Verify Class Configuration

The `classes.txt` file should contain only your target classes:

```
person
cell phone
book
```

This tells the system which objects to detect and label. The file maps COCO model's 80 classes to your 3-class list.

### STEP 2: Capture Reference Images

The system needs reference images to calculate focal length. Use `CaptureReferenceImage.py`:

```bash
python CaptureReferenceImage.py
```

**Procedure:**
1. Position a **person** at exactly 45 inches from camera
2. Press `'c'` to capture image → saves as `image14.png`
3. Position a **cell phone** at exactly 45 inches
4. Press `'c'` to capture image → saves as `image4.png` (or later images)
5. Press `'q'` to quit

**Why 45 inches?**
This is the `KNOWN_DISTANCE` constant. It's your calibration distance. You can change it, but must measure accurately!

### STEP 3: Verify Reference Images

Check that the reference images were saved correctly:

```bash
ls ReferenceImages/
# Should show image4.png, image14.png, etc.
```

Open them to ensure:
- Object is clearly visible
- Object is the main focus
- Distance was measured correctly (45 inches)

### STEP 4: Run Distance Estimation

The main script: `DistanceEstimation.py`

```bash
python DistanceEstimation.py
```

**Controls:**
- Webcam window opens showing detections
- Distance displayed in inches above each object
- Alert sound triggers when object is < 30 inches
- Press `'q'` to quit

**What happens:**
1. Loads reference images (image14.png for person, image4.png for phone)
2. Calculates focal length for each object type
3. Starts webcam (camera index 0)
4. For each frame:
   - Detects objects with YOLO
   - Filters only person, cell phone, book
   - Calculates distance using formula
   - Draws bounding box + label with distance
   - If distance < 30 inches → audio alert

### STEP 5: (Optional) Record Videos

Use `demo_Recoder.py` to capture video with detections:

```bash
python demo_Recoder.py
```

- Press `'c'` to start/stop capturing
- Video saved as `out.mp4`
- Press `'q'` to quit

---

## Code Explanation

### 1. classes.txt Loading

```python
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
```

Reads the 3 class names. These correspond to indices 0, 1, 2 in the list.

### 2. Class Filtering with COCO Mapping

YOLOv8's pre-trained model (yolov8n.pt) knows 80 COCO classes. We only want 3:

```python
# COCO class IDs for our targets:
ALLOWED_CLASS_IDS = {0, 67, 73}  # person=0, cell phone=67, book=73
CLASS_ID_MAP = {0: 0, 67: 1, 73: 2}  # Maps COCO ID → our class_names index
```

**Why this mapping?**
- COCO model returns class_id (0-79)
- We filter: if `class_id` not in `{0, 67, 73}`, skip it
- We map COCO ID to our 3-element `class_names` list:
  - COCO 0 (person) → class_names[0]
  - COCO 67 (cell phone) → class_names[1]
  - COCO 73 (book) → class_names[2]

### 3. YOLO Model Loading

```python
yoloNet = YOLO('yolov8n.pt')
```

- `yolov8n.pt` is the "nano" version (smallest, fastest)
- Auto-downloads on first run (~6MB)
- Loads once at startup, reused for all frames

### 4. Object Detection Function

```python
def object_detector(image):
    results = yoloNet(image, verbose=False)[0]
    data_list = []
    for result in results.boxes.data:
        x1, y1, x2, y2, score, class_id = result.tolist()
        # ... filter and process
```

- `results.boxes.data` contains all detections
- Each detection: `[x1, y1, x2, y2, confidence, class_id]`
- Coordinates are in pixels (top-left to bottom-right)

### 5. Focal Length Calculation

```python
def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length
```

**Example:**
- Person at 45 inches → width = 200 pixels in reference image
- Real person width = 16 inches (average shoulder width)
- Focal length = (200 × 45) / 16 = 562.5 pixels

This focal length is **property of your camera** - it stays constant!

### 6. Distance Calculation

```python
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance
```

**Example:**
- Focal length = 562.5 (from calibration)
- Real person width = 16 inches
- Person width in current frame = 150 pixels
- Distance = (16 × 562.5) / 150 = 60 inches

### 7. Confidence Threshold

```python
CONFIDENCE_THRESHOLD = 0.2  # 20% confidence minimum
```

YOLO returns a confidence score (0 to 1). Filters out weak detections.

### 8. Audio Alert System

```python
def sound():
    engine.say("Object very close")
    engine.runAndWait()
```

Uses `pyttsx3` for text-to-speech. Runs in a separate thread to avoid blocking video.

---

## Customization

### Change Target Classes

Edit `classes.txt`:

```
person
cell phone
book
```

**Important:** Must match the COCO model's class names exactly:
- "person" ✓
- "cell phone" ✓ (with space)
- "book" ✓
- Not "mobile" or "smartphone"

To add more classes, you need to:
1. Add the class name to `classes.txt`
2. Add its COCO class ID to `ALLOWED_CLASS_IDS` and `CLASS_ID_MAP`
3. Update distance estimation logic (add focal length constants)

### Adjust Distance Constants

```python
# In DistanceEstimation.py
KNOWN_DISTANCE = 45  # Calibration distance in inches
PERSON_WIDTH = 16    # Average person shoulder width in inches
MOBILE_WIDTH = 3.0   # Average phone width in inches
```

**For books:**
Add a constant:
```python
BOOK_WIDTH = 8  # Average book width in inches (adjust as needed)
```

Then add distance calculation:
```python
elif obj_name == 'book':
    distance = distance_finder(focal_book, BOOK_WIDTH, width_in_frame)
```

### Change Camera

```python
cap = cv.VideoCapture(0)  # Change 0 to:
# 1, 2, ... for different cameras
# Or use camera URL: cv.VideoCapture('http://ip:port/video')
```

### Change Alert Distance

```python
if distance < 30.0:  # Alert threshold in inches
    # trigger alert
```

### Adjust Confidence Threshold

```python
CONFIDENCE_THRESHOLD = 0.2  # Lower = more detections, more false positives
                           # Higher = fewer but more confident detections
```

### Use Different YOLO Model

```python
yoloNet = YOLO('yolov8s.pt')  # Small (more accurate, slower)
yoloNet = YOLO('yolov8m.pt')  # Medium
yoloNet = YOLO('yolov8l.pt')  # Large
yoloNet = YOLO('yolov8x.pt')  # X-Large (most accurate, slowest)
```

Models auto-download on first run. Sizes:
- `yolov8n.pt`: ~6MB
- `yolov8s.pt`: ~22MB
- `yolov8m.pt`: ~52MB
- `yolov8l.pt`: ~88MB
- `yolov8x.pt`: ~131MB

---

## Troubleshooting

### 1. "ModuleNotFoundError: No module named 'ultralytics'"

```bash
pip install ultralytics
```

### 2. Webcam not opening

- Check if another program is using the camera
- Try different camera index: `cv.VideoCapture(1)`
- On some systems, may need to use `cv.CAP_DSHOW` on Windows:
  ```python
  cap = cv.VideoCapture(0, cv.CAP_DSHOW)
  ```

### 3. Reference image detection fails

- Ensure object is clearly visible and centered
- Object should fill reasonable portion of frame
- Check that `ReferenceImages/` folder exists
- Verify `KNOWN_DISTANCE` matches actual measurement

### 4. Inaccurate distance estimates

- Ensure reference image was taken at EXACT `KNOWN_DISTANCE`
- Real object width constants may need adjustment for your use case
- Distance formula assumes object is perpendicular to camera
- Accuracy decreases for very far objects (> 10 feet)

### 5. Book not detected

- YOLO was primarily trained on person/phone, books may have lower accuracy
- Try adjusting `CONFIDENCE_THRESHOLD` lower (e.g., 0.15)
- Ensure book is clearly visible, spine facing camera for better detection

### 6. Audio alert not working

- Ensure `pyttsx3` is installed
- On Linux, may need `espeak`: `sudo apt-get install espeak`
- Audio may be blocked by system permissions

### 7. Classes mismatch errors

The classes.txt must have EXACTLY 3 lines matching:
```
person
cell phone
book
```

No leading/trailing spaces. No blank line at end.

---

## Understanding the Numbers

### Focal Length
- Your camera's "magnification factor"
- Calculated once during calibration
- Typical range: 300-2000 pixels
- Example: iPhone focal length ≈ 1200-1500 pixels

### Real Width Constants
- **PERSON_WIDTH = 16 inches**: Average adult shoulder width
- **MOBILE_WIDTH = 3.0 inches**: Typical smartphone width (iPhone ~2.8", Galaxy ~3.0")
- **BOOK_WIDTH**: For paperbacks ~7-8", hardcovers ~9-10"

These are approximations. For precise work, measure actual objects.

### Expected Accuracy
- **Excellent** (< 5% error): 1-5 feet distance
- **Good** (5-10% error): 5-10 feet
- **Fair** (10-20% error): 10-15 feet
- **Poor** (> 20%): > 15 feet

Accuracy depends on:
- Good reference calibration
- Object facing perpendicular to camera
- Accurate real-width measurement
- YOLO detection accuracy

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│                     Main Loop                           │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │   Camera    │───▶│   YOLO       │───▶│ Filter   │ │
│  │   Frame     │    │  Detection   │    │  Classes │ │
│  └─────────────┘    └──────────────┘    └──────────┘ │
│                               │                        │
│                               ▼                        │
│                   ┌─────────────────────┐             │
│                   │ Distance Calculator │             │
│                   │  (Formula-based)    │             │
│                   └─────────────────────┘             │
│                               │                        │
│                               ▼                        │
│                 ┌─────────────────────────┐           │
│                 │  Alert & Display Logic │           │
│                 └─────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

---

## Advanced: Training Custom Model

Current setup uses COCO pre-trained model. To train on YOUR custom objects:

1. Prepare dataset (images + annotations in YOLO format)
2. Create `data.yaml` with class names
3. Train:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.yaml')
   model.train(data='custom_data.yaml', epochs=100)
   ```
4. Use trained weights: `yoloNet = YOLO('best.pt')`

---

## Summary

This project demonstrates:
- **Real-time object detection** with YOLOv8
- **Distance estimation** using camera calibration
- **Practical computer vision** applications
- **Filtering and mapping** class IDs
- **Audio-visual feedback** systems

**Key Takeaway:** Distance estimation is possible because we know:
1. Camera focal length (from calibration)
2. Real object size (from measurements)
3. Object size in pixels (from YOLO detection)

---

## Resources

- YOLOv8 Documentation: https://docs.ultralytics.com/
- OpenCV Tutorials: https://docs.opencv.org/
- Pinhole Camera Model: https://en.wikipedia.org/wiki/Pinhole_camera_model
- COCO Dataset Classes: https://cocodataset.org/#explore

---

**Created by:** Mayuresh Vengurlekar
**Date:** March 2025
**Version:** 1.0
