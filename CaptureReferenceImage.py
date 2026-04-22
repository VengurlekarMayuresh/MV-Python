import cv2 as cv
import time
from ultralytics import YOLO
import os

CONFIDENCE_THRESHOLD = 0.5

COLORS = [(0,255,255),(255,255,0),(0,255,0),(255,0,0)]
fonts = cv.FONT_HERSHEY_COMPLEX

# Load class names
with open("classes.txt", "r") as f:
    class_names = [c.strip() for c in f.readlines()]

# COCO IDs
CLASS_ID_MAP = {0: 0, 67: 1, 73: 2}
ALLOWED_CLASS_IDS = {0, 67, 73}

# Load YOLO model
yoloNet = YOLO('yolov8n.pt')

def ObjectDetector(image):
    results = yoloNet(image, verbose=False)[0]
    detections = []

    for result in results.boxes.data:
        x1, y1, x2, y2, score, class_id = result.tolist()
        class_id = int(class_id)

        if class_id not in ALLOWED_CLASS_IDS:
            continue
        if score < CONFIDENCE_THRESHOLD:
            continue

        class_idx = CLASS_ID_MAP[class_id]
        w, h = int(x2 - x1), int(y2 - y1)

        detections.append((class_names[class_idx], w, h))

        color = COLORS[class_idx % len(COLORS)]
        label = f"{class_names[class_idx]}: {round(score,2)}"

        cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv.putText(image, label, (int(x1), int(y1)-10), fonts, 0.5, color, 2)

    return detections


# STEP 1: Distance input
try:
    dist = input("Enter distance (in inches): ").strip()
    KNOWN_DISTANCE = float(dist) if dist else 30.0
except:
    KNOWN_DISTANCE = 30.0

# Save distance
with open("calibration_data.txt", "w") as f:
    f.write(str(KNOWN_DISTANCE))

print(f"Distance set to {KNOWN_DISTANCE} inches.")
print("Press 'p' = person | 'm' = mobile | 'q' = quit")

camera = cv.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    original = frame.copy()
    detections = ObjectDetector(frame)

    cv.imshow('frame', frame)
    key = cv.waitKey(1)

    # PERSON CAPTURE
    if key == ord('p'):
        person = [d for d in detections if d[0] == 'person']

        if person:
            width = person[0][1]

            cv.imwrite('ReferenceImages/person_ref.png', original)

            with open("person_width.txt", "w") as f:
                f.write(str(width))

            # Save the calibration distance specifically for person
            with open("person_calibration_dist.txt", "w") as f:
                f.write(str(KNOWN_DISTANCE))

            print(f"Person captured | width: {width}px | at distance: {KNOWN_DISTANCE} inches")
        else:
            print("No person detected!")

    # MOBILE CAPTURE
    if key == ord('m'):
        mobile = [d for d in detections if d[0] == 'cell phone']

        if mobile:
            width = mobile[0][1]

            cv.imwrite('ReferenceImages/mobile_ref.png', original)

            with open("mobile_width.txt", "w") as f:
                f.write(str(width))

            # Save the calibration distance specifically for mobile
            with open("mobile_calibration_dist.txt", "w") as f:
                f.write(str(KNOWN_DISTANCE))

            print(f"Mobile captured | width: {width}px | at distance: {KNOWN_DISTANCE} inches")
        else:
            print("No mobile detected!")

    if key == ord('q'):
        break

cv.destroyAllWindows()
camera.release()