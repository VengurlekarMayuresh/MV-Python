import cv2 as cv
import time
from ultralytics import YOLO

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
# Mapping from COCO ID to our class_names index
CLASS_ID_MAP = {0: 0, 67: 1, 73: 2}
ALLOWED_CLASS_IDS = {0, 67, 73}

# Load YOLOv8 model (latest)
yoloNet = YOLO('yolov8n.pt')  # Uses yolov8n.pt - will auto-download

model = yoloNet  # Direct use of YOLO model

# setting camera


def ObjectDetector(image):
    # YOLOv8 inference
    results = yoloNet(image, verbose=False)[0]

    for result in results.boxes.data:
        x1, y1, x2, y2, score, class_id = result.tolist()
        class_id = int(class_id)

        # Filter: only process person, cell phone, and book
        if class_id not in ALLOWED_CLASS_IDS:
            continue

        # Map COCO class ID to our class_names index
        class_idx = CLASS_ID_MAP[class_id]

        if score < CONFIDENCE_THRESHOLD:
            continue

        # Convert to (x, y, w, h) format
        w = int(x2 - x1)
        h = int(y2 - y1)
        box = (int(x1), int(y1), w, h)

        color = COLORS[class_idx % len(COLORS)]
        label = "%s : %f" % (class_names[class_idx], score)
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-10), fonts, 0.5, color, 2)


camera = cv.VideoCapture(0)
counter = 0
capture = False
number = 0
while True:
    ret, frame = camera.read()

    orignal = frame.copy()
    ObjectDetector(frame)
    cv.imshow('oringal', orignal)

    print(capture == True and counter < 10)
    if capture == True and counter < 10:
        counter += 1
        cv.putText(
            frame, f"Capturing Img No: {number}", (30, 30), fonts, 0.6, PINK, 2)
    else:
        counter = 0

    cv.imshow('frame', frame)
    key = cv.waitKey(1)

    if key == ord('c'):
        capture = True
        number += 1
        cv.imwrite(f'ReferenceImages/image{number}.png', orignal)
    if key == ord('q'):
        break
cv.destroyAllWindows()
