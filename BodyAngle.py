import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline

# Initial Position Of FootPad
STATIONARY_POINT = []
BASE_POINT = []
tip_pos = []
angle = []
time = []


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        STATIONARY_POINT.append(x)
        STATIONARY_POINT.append(y)

    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, ' ', y)
        BASE_POINT.append(x)
        BASE_POINT.append(y)


def graphData():
    x = np.array(time)
    y = np.array(angle)
    xnew = np.linspace(x.min(), x.max(), 200)

    # define spline
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(xnew)

    # create smooth line chart
    plt.plot(xnew, y_smooth)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (Degree)")
    plt.show()


def deflection_angle(pt2):
    # Base Slope
    vector1 = [BASE_POINT[0] - STATIONARY_POINT[0], BASE_POINT[1] - STATIONARY_POINT[1]]
    vector2 = [pt2[0] - STATIONARY_POINT[0], pt2[1] - STATIONARY_POINT[1]]

    norm_1 = np.sqrt(pow(vector1[0], 2) + pow(vector1[1], 2))
    norm_2 = np.sqrt(pow(vector2[0], 2) + pow(vector2[1], 2))
    a_dot_b = (vector1[0] * vector2[0]) + (vector1[1] * vector2[1])

    cos_theta = np.arccos(a_dot_b / (norm_1 * norm_2))
    # print(norm_line1, norm_line2, a_dot_b, cos_theta)
    angle.append(cos_theta*180/np.pi)


def main(source):
    time_counter = 0
    cap = cv2.VideoCapture(source)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (540, 380), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            # Crop image so that contour does not detect other edges
            frame = frame[0:540, 0:380]
            # cv2.imshow('image', frame)
            # cv2.waitKey(0)

            if time_counter == 0:
                cv2.imshow('image', frame)
                cv2.setMouseCallback('image', click_event)
                cv2.waitKey(0)

            # Preprocessing image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, thresh2 = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow('image', thresh2)
            # cv2.waitKey(0)
            medianFiltered = cv2.medianBlur(thresh2, 5)

            # Detecting contour
            contours, hierarchy = cv2.findContours(medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            copiedImage = np.copy(frame)
            c = max(contours, key=cv2.contourArea)

            # Finding the top extreme point in contour
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            cv2.circle(copiedImage, extTop, 8, (255, 0, 0), -1)

            deflection_angle(extTop)
            time.append(time_counter)

            cv2.imshow('Image', copiedImage)
            time_counter += 1 / FPS

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Convert the arrays into csv files
    arr = np.asarray([time, angle], dtype=object, )
    pd.DataFrame(arr).to_csv('ph-7.csv')


if __name__ == "__main__":
    main('screen-capture (41)_working.mp4')
    graphData()
