import numpy as np
import cv2 as cv
from PIL import Image
from utils import get_limits

yellow = [0, 255, 255]  # BGR formatında sarı
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame alınamadı!")
        break

    hsvImage = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(yellow)

    mask = cv.inRange(hsvImage, lowerLimit, upperLimit)

    mask_ = Image.fromarray(mask)

    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()