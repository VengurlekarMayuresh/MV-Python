import cv2 as cv
import numpy as np
from ultralytics import YOLO
import threading
import pyttsx3
import os

engine = pyttsx3.init()

# Load Calibration Data
if os.path.exists("calibration_data.txt"):
    with open("calibration_data.txt", "r") as f:
        KNOWN_DISTANCE = float(f.read().strip())
else:
    KNOWN_DISTANCE = 30.0  # Default if file missing

ALERT_DISTANCE = 25.0  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES

# Object detector constant
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# COCO class IDs mapping
ALLOWED_CLASS_IDS = {0: 0, 67: 1, 73: 2}

# Load YOLOv8 model
yoloNet = YOLO('yolov8n.pt')

def object_detector(image):
    results = yoloNet(image, verbose=False)[0]
    data_list = []
    for result in results.boxes.data:
        x1, y1, x2, y2, score, class_id = result.tolist()
        class_id = int(class_id)
        if class_id not in ALLOWED_CLASS_IDS: continue
        class_idx = ALLOWED_CLASS_IDS[class_id]
        if score < CONFIDENCE_THRESHOLD: continue
        width = int(x2 - x1)
        height = int(y2 - y1)
        box = (int(x1), int(y1), width, height)
        color = COLORS[class_idx % len(COLORS)]
        label = "%s : %f" % (class_names[class_idx], score)
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)
        data_list.append([class_names[class_idx], width, (box[0], box[1] - 2)])
    return data_list

def focal_length_finder(measured_distance, real_width, width_in_rf):
    return (width_in_rf * measured_distance) / real_width

def distance_finder(focal_length, real_object_width, width_in_frame):
    return (real_object_width * focal_length) / width_in_frame

is_speaking = False
def sound():
    global is_speaking
    if is_speaking: return
    is_speaking = True
    try:
        engine.say("Object very close")
        engine.runAndWait()
    except: pass
    is_speaking = False

# Automatic Calibration using saved references
try:
    ref_person = cv.imread('ReferenceImages/person_ref.png')
    person_data = object_detector(ref_person)
    person_width_in_rf = person_data[0][1]
    focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
    print("Person calibration successful.")
except:
    focal_person = 800.0
    print("Person calibration failed, using default.")

try:
    ref_mobile = cv.imread('ReferenceImages/mobile_ref.png')
    mobile_data = object_detector(ref_mobile)
    mobile_width_in_rf = mobile_data[0][1]
    focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
    print("Mobile calibration successful.")
except:
    focal_mobile = 800.0
    print("Mobile calibration failed, using default.")

print(f"Using Calibration Distance: {KNOWN_DISTANCE} inches")

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    data = object_detector(frame)
    for d in data:
        obj_name, width_in_frame, (x, y) = d
        if obj_name == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, width_in_frame)
        elif obj_name == 'cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, width_in_frame)
        else:
            distance = distance_finder(800.0, 15.0, width_in_frame)

        if distance < ALERT_DISTANCE:
            threading.Thread(target=sound).start()
            cv.putText(frame, "ALERT: Object Very Close!", (x, y - 20), FONTS, 0.6, (0, 0, 255), 2)
        
        cv.rectangle(frame, (x, y-3), (x+150, y+23), BLACK, -1)
        cv.putText(frame, f'Dis: {round(distance, 2)} inch', (x+5, y+13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'): break

cv.destroyAllWindows()
cap.release()
