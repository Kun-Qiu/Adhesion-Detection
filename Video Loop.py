import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    cap = cv2.VideoCapture("screen-capture (41)_working.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (540, 380), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('image', frame)
        cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


