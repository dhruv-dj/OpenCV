import numpy as np
import cv2

cap = cv2.VideoCapture(0)

def make_720p():
    cap.set(3,1280)
    cap.set(4,720)

def make_1080p():
    cap.set(3,1920)
    cap.set(4,1080)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while True:
    ret, frame = cap.read()
    frame2= rescale_frame(frame,150)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame2)
    cv2.imshow('gray', gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()