# Technical Manual: Line-by-Line Code Breakdown

This document provides a literal line-by-line explanation of the core scripts in the project.

---

## 1. DistanceEstimation.py (The Main App)

| Line # | Code | Detailed Explanation |
| :--- | :--- | :--- |
| 1 | `import cv2 as cv` | Imports OpenCV, the primary library for computer vision and image processing. |
| 2 | `import numpy as np` | Imports NumPy for mathematical operations on arrays (images). |
| 3 | `from ultralytics import YOLO` | Imports the YOLO class from Ultralytics to load the object detection model. |
| 4 | `import threading` | Imports threading to run the voice alert in the background without freezing the video. |
| 5 | `import pyttsx3` | Imports the Text-to-Speech library to convert text to audible alerts. |
| 6 | `import os` | Imports OS for checking if files (like calibration data) exist on the disk. |
| 8 | `engine = pyttsx3.init()` | Initializes the speech engine to prepare for voice alerts. |
| 11 | `if os.path.exists("calibration_data.txt"):` | Checks if we have previously saved the calibration distance from the first script. |
| 12 | `with open(...) as f:` | Opens the calibration file in "read" mode. |
| 13 | `KNOWN_DISTANCE = float(...)` | Reads the distance (e.g., 30.0) and converts it to a decimal number. |
| 15 | `KNOWN_DISTANCE = 30.0` | If the file is missing, we use a default of 30 inches as a fallback. |
| 17 | `ALERT_DISTANCE = 25.0` | Sets the threshold (25 inches). If an object is closer, an alert triggers. |
| 18 | `PERSON_WIDTH = 16` | The average width of a human shoulder-to-shoulder in inches. |
| 19 | `MOBILE_WIDTH = 3.0` | The average width of a standard mobile phone in inches. |
| 22 | `CONFIDENCE_THRESHOLD = 0.2` | The AI must be at least 20% certain to display the detection box. |
| 23 | `NMS_THRESHOLD = 0.3` | Used to remove "duplicate" boxes around the same object (Non-Maximum Suppression). |
| 26 | `COLORS = [...]` | A list of BGR colors used to draw boxes for different object types. |
| 27 | `GREEN = (0, 255, 0)` | A standard BGR color for green text. |
| 29 | `FONTS = cv.FONT_HERSHEY_COMPLEX` | Selects a clean, bold font style for the text on the screen. |
| 32 | `class_names = []` | Initializes an empty list to hold the names of objects. |
| 33 | `with open("classes.txt", "r") as f:` | Opens the file containing names like "person", "cell phone". |
| 34 | `class_names = [cname.strip()...]` | Cleans up the text and stores each name in the list. |
| 37 | `ALLOWED_CLASS_IDS = {0: 0, 67: 1, 73: 2}` | Maps YOLO's ID (0=Person) to our internal list index. |
| 40 | `yoloNet = YOLO('yolov8n.pt')` | Loads the YOLOv8 "Nano" model (the trained AI file). |
| 42 | `def object_detector(image):` | Defines the function that will analyze every video frame. |
| 43 | `results = yoloNet(image...)[0]` | Sends the image to the AI and gets the detection results. |
| 45 | `for result in results.boxes.data:` | Loops through every object the AI found in the image. |
| 46 | `x1, y1, x2, y2, score, class_id = ...` | Extracts coordinates, confidence score, and object type. |
| 51-52 | `width = int(x2 - x1)` | Calculates the width of the bounding box in pixels. |
| 56 | `cv.rectangle(image, box, color, 2)` | Draws a colored box around the detected object on the frame. |
| 58 | `data_list.append(...)` | Saves the object name and pixel width to a list for calculation. |
| 61 | `def focal_length_finder(...)` | Formula to calculate how the camera "sees" distance during calibration. |
| 64 | `def distance_finder(...)` | The real-time formula to calculate distance in inches based on pixel width. |
| 68 | `def sound():` | The function that handles the audible "Object very close" warning. |
| 71 | `is_speaking = True` | Prevents the computer from saying the same alert 100 times at once. |
| 73 | `engine.say(...)` | Queues the text message for speech. |
| 74 | `engine.runAndWait()` | Actually plays the sound through the speakers. |
| 80 | `ref_person = cv.imread(...)` | Opens the calibration photo of a person taken earlier. |
| 81 | `person_data = object_detector(ref_person)` | Analyzes the calibration photo to see how many pixels wide the person was. |
| 83 | `focal_person = focal_length_finder(...)` | Calculates the Focal Length for detecting people. |
| 101 | `cap = cv.VideoCapture(0)` | Opens your computer's webcam (0 is usually the default camera). |
| 102 | `while True:` | Starts a continuous loop that runs as long as the app is open. |
| 103 | `ret, frame = cap.read()` | "Grab" one frame of video from the camera. |
| 105 | `data = object_detector(frame)` | Detects all objects in the current live video frame. |
| 109 | `distance = distance_finder(...)` | Calculates the distance of a person using the focal length. |
| 115 | `if distance < ALERT_DISTANCE:` | Checks if the person is too close (under 25 inches). |
| 116 | `threading.Thread(target=sound).start()` | Plays the voice alert in the background (asynchronously). |
| 119 | `cv.rectangle(frame, ...)` | Draws a small black box behind the distance text for readability. |
| 120 | `cv.putText(frame, f'Dis: {round...}')` | Displays the final distance on the screen. |
| 122 | `cv.imshow('frame', frame)` | Shows the final window with the video and detections to the user. |
| 123 | `if cv.waitKey(1) == ord('q'): break` | If you press 'q' on your keyboard, the app closes. |

---

## 2. CaptureReferenceImage.py (The Calibration Tool)

| Line # | Code | Detailed Explanation |
| :--- | :--- | :--- |
| 46 | `dist = input(...)` | Asks the user to type in how far away they are sitting (e.g., 30). |
| 53 | `with open("calibration_data.txt", "w")...` | Writes that distance into a file so the other script can remember it. |
| 67 | `original = frame.copy()` | Saves a "clean" version of the frame without any boxes drawn on it. |
| 76 | `if key == ord('p'):` | Listens for the 'P' key to capture a person's reference photo. |
| 77 | `cv.imwrite('...', original)` | Saves the clean frame as a PNG image file in the ReferenceImages folder. |
| 79 | `msg_timer = time.time() + 2` | Sets a timer to show a "Captured!" message on screen for 2 seconds. |
