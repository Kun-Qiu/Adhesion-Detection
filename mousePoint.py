import cv2
import numpy as np
import pandas as pd

time = []
x_pos = []
y_pos = []

X_pos = []
Y_pos = []


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks --> To determine the distance
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        x_pos.append(x)
        y_pos.append(y)
        # time.append(time_counter)
        cv2.imshow('image', frame)

    # # checking for right mouse clicks --> to determine the deflection angle
    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, ' ', y)
        X_pos.append(x)
        Y_pos.append(x)
        cv2.imshow('image', frame)


# driver function
if __name__ == "__main__":
    cap = cv2.VideoCapture("0_720.mp4")
    FPS = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        time_counter = 0
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (540, 380), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            cv2.imshow('image', frame)
            cv2.setMouseCallback('image', click_event)
            # cv2.waitKey(0)

            time_counter += 1 / FPS

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    arr = np.asarray([x_pos, y_pos, X_pos, Y_pos], dtype=object, )
    pd.DataFrame(arr).to_csv('vid1.csv')
