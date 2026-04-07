import cv2 as cv
import numpy as np
from ultralytics import YOLO
import threading
import pyttsx3

engine = pyttsx3.init()

# Distance constants
KNOWN_DISTANCE = 45  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES

# Object detector constant
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# COCO class IDs for our target classes: person=0, cell phone=67, book=73
ALLOWED_CLASS_IDS = {0: 0, 67: 1, 73: 2}  # COCO ID -> index in class_names

# Load YOLOv8 model (latest)
yoloNet = YOLO('yolov8n.pt')  # Use yolov8n.pt - download automatically

# object detector function / method
def object_detector(image):
    # YOLOv8 inference
    results = yoloNet(image, verbose=False)[0]

    data_list = []
    for result in results.boxes.data:
        x1, y1, x2, y2, score, class_id = result.tolist()
        class_id = int(class_id)

        # Filter: only process person, cell phone, and book
        if class_id not in ALLOWED_CLASS_IDS:
            continue

        # Map COCO class ID to our class_names index
        class_idx = ALLOWED_CLASS_IDS[class_id]

        if score < CONFIDENCE_THRESHOLD:
            continue

        # Convert to xywh format (x, y, width, height) for compatibility
        width = int(x2 - x1)
        height = int(y2 - y1)
        box = (int(x1), int(y1), width, height)

        color = COLORS[class_idx % len(COLORS)]

        label = "%s : %f" % (class_names[class_idx], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        data_list.append([class_names[class_idx], width, (box[0], box[1] - 2)])

        # returning list containing the object data.
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance


is_speaking = False
def sound():
    global is_speaking
    if is_speaking: return
    is_speaking = True
    try:
        engine.say("Object very close")
        engine.runAndWait()
    except:
        pass
    is_speaking = False
    return 0


try:
    # reading the reference image from dir
    ref_person = cv.imread('ReferenceImages/image14.png')
    person_data = object_detector(ref_person)
    person_width_in_rf = person_data[0][1]
    focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
except:
    person_width_in_rf = 200
    focal_person = 800.0  # Default focal length

try:
    ref_mobile = cv.imread('ReferenceImages/image4.png')
    mobile_data = object_detector(ref_mobile)
    # the second object might be the cell phone, but let's just grab the first one if it exists
    mobile_width_in_rf = mobile_data[0][1]
    focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
except:
    mobile_width_in_rf = 100
    focal_mobile = 800.0

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    data = object_detector(frame)
    for d in data:
        obj_name = d[0]
        width_in_frame = d[1]
        x, y = d[2]

        if obj_name == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, width_in_frame)
        elif obj_name == 'cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, width_in_frame)
        else:
            # Generic width of 15 inches for unknown objects
            distance = distance_finder(800.0, 15.0, width_in_frame)

        if distance < 30.0:  # Alert when object is within 30 inches
            soundThread = threading.Thread(target=sound)
            soundThread.start()
            cv.putText(frame, "ALERT: Object Very Close!", (x, y - 20), FONTS, 0.6, (0, 0, 255), 2)
        cv.rectangle(frame, (x, y-3), (x+150, y+23), BLACK, -1)
        cv.putText(frame, f'Dis: {round(distance, 2)} inch', (x+5, y+13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows()
cap.release()
