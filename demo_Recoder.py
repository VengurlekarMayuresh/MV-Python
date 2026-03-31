import cv2 as cv
import numpy as np
from ultralytics import YOLO

# Distance constants
KNOWN_DISTANCE = 45  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
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

# Load YOLOv8 model (latest)
yoloNet = YOLO('yolov8n.pt')  # Uses yolov8n.pt - will auto-download

# object detector function / method
def object_detector(image):
    # YOLOv8 inference
    results = yoloNet(image, verbose=False)[0]

    data_list = []
    for result in results.boxes.data:
        x1, y1, x2, y2, score, class_id = result.tolist()
        class_id = int(class_id)

        if score < CONFIDENCE_THRESHOLD:
            continue

        # Convert to xywh format (x, y, width, height)
        width = int(x2 - x1)
        height = int(y2 - y1)
        box = (int(x1), int(y1), width, height)

        color = COLORS[class_id % len(COLORS)]

        label = "%s : %f" % (class_names[class_id], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        data_list.append([class_names[class_id], width, (box[0], box[1] - 2)])
        # returning list containing the object data.
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance


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
    mobile_width_in_rf = mobile_data[1][1]
    focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
except:
    mobile_width_in_rf = 100
    focal_mobile = 800.0

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

cap = cv.VideoCapture(3)
fourcc = cv.VideoWriter_fourcc(*'XVID')
Recoder = cv.VideoWriter('out.mp4', fourcc, 8.0, (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
while True:
    ret, frame = cap.read()

    data = object_detector(frame)
    for d in data:
        if d[0] == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        cv.rectangle(frame, (x, y-3), (x+150, y+23), BLACK, -1)
        cv.putText(frame, f'Dis: {round(distance, 2)} inch', (x+5, y+13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame', frame)
    Recoder.write(frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows()
Recoder.release()
cap.release()
