import cv2 as cv
import time
from ultralytics import YOLO
import os

# setting parameters
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5

# colors for object detected
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
RED = (0, 0, 255)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)
fonts = cv.FONT_HERSHEY_COMPLEX

# reading class name from text file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# COCO class IDs for our target classes: person=0, cell phone=67, book=73
CLASS_ID_MAP = {0: 0, 67: 1, 73: 2}
ALLOWED_CLASS_IDS = {0, 67, 73}

# Load YOLOv8 model
yoloNet = YOLO('yolov8n.pt')

def ObjectDetector(image):
    results = yoloNet(image, verbose=False)[0]
    for result in results.boxes.data:
        x1, y1, x2, y2, score, class_id = result.tolist()
        class_id = int(class_id)
        if class_id not in ALLOWED_CLASS_IDS: continue
        class_idx = CLASS_ID_MAP[class_id]
        if score < CONFIDENCE_THRESHOLD: continue
        w, h = int(x2 - x1), int(y2 - y1)
        box = (int(x1), int(y1), w, h)
        color = COLORS[class_idx % len(COLORS)]
        label = "%s : %f" % (lass_names[class_idx], score)
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-10), fonts, 0.5, color, 2)
c
# STEP 1: Ask for distance here only
try:
    dist = input("Enter the distance you are sitting at (in inches): ").strip()
    KNOWN_DISTANCE = float(dist) if dist else 30.0
except:
    KNOWN_DISTANCE = 30.0

# Save distance for DistanceEstimation.py
with open("calibration_data.txt", "w") as f:
    f.write(str(KNOWN_DISTANCE))

print(f"Distance set to {KNOWN_DISTANCE} inches. Calibration data saved.")
print("Controls: \n 'p' - Capture Person \n 'm' - Capture Mobile \n 'q' - Quit")

camera = cv.VideoCapture(0)
capture_msg = ""
msg_timer = 0

while True:
    ret, frame = camera.read()
    if not ret: break
    
    original = frame.copy()
    ObjectDetector(frame)
    
    if time.time() < msg_timer:
        cv.putText(frame, capture_msg, (30, 50), fonts, 0.7, PINK, 2)

    cv.imshow('frame', frame)
    key = cv.waitKey(1)

    if key == ord('p'):
        cv.imwrite('ReferenceImages/person_ref.png', original)
        capture_msg = "Captured Person Reference!"
        msg_timer = time.time() + 2
        print(capture_msg)
    
    if key == ord('m'):
        cv.imwrite('ReferenceImages/mobile_ref.png', original)
        capture_msg = "Captured Mobile Reference!"
        msg_timer = time.time() + 2
        print(capture_msg)

    if key == ord('q'):
        break

cv.destroyAllWindows()
camera.release()
