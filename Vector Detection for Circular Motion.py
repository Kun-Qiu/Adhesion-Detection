import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline


# STATIONARY_POINT = []


def vectorFunction(extPt, midPt):
    vector = [extPt[0] - midPt[0], extPt[1] - midPt[1]]
    return vector


def findMaxDistance(enum_list, midpoint):
    max_distance = 0
    key = 0

    # Finding max distance
    for arg1, arg2 in enum_list:
        vector = vectorFunction(arg2, midpoint)
        distance = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
        if distance > max_distance:
            max_distance = distance
            key = arg1
    return key


# def click_event(event, x, y, flags, params):
#     # checking for left mouse clicks
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, ' ', y)
#         STATIONARY_POINT.append(x)
#         STATIONARY_POINT.append(y)


def processData(source):
    vector_x = []
    vector_y = []
    Time = []
    distance = []
    time_counter = 0

    cap = cv2.VideoCapture(source)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (540, 380), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

            # if time_counter == 0:
            #     cv2.imshow('image', frame)
            #     cv2.setMouseCallback('image', click_event)
            #     cv2.waitKey(0)

            # Preprocessing image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Uncomment this line if using it to find probe angle
            ret, thresh2 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

            # Uncomment this line if using it to find magnetic field angle
            # ret, thresh2 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

            # cv2.imshow('image', thresh2)
            # cv2.imshow('og', frame)
            # cv2.waitKey(0)
            medianFiltered = cv2.medianBlur(thresh2, 5)

            # Detecting contour
            contours, hierarchy = cv2.findContours(medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            copiedImage = np.copy(frame)
            c = max(contours, key=cv2.contourArea)

            # Spin Disk Direction
            M = cv2.moments(c)

            if M["m00"] != 0:
                center_X = int(M["m10"] / M["m00"])
                center_Y = int(M["m01"] / M["m00"])
            else:
                center_X, center_Y = 0, 0

            midpoint = [center_X, center_Y]

            # Finding the extreme points in contour
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBottom = tuple(c[c[:, :, 1].argmax()][0])
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])

            extreme_points = [extTop, extBottom, extLeft, extRight]

            # Add distance to the array
            max_distance_key = findMaxDistance(enumerate(extreme_points), midpoint)
            for key, val in enumerate(extreme_points):
                if max_distance_key == key:
                    vector = vectorFunction(val, midpoint)
                    distance.append(np.sqrt(vector[0] ** 2 + vector[1] ** 2))
                    vector_x.append(vector[0])
                    vector_y.append(vector[1])

                    cv2.circle(copiedImage, extreme_points[key], 8, (255, 0, 0), -1)
                    print(np.sqrt(vector[0] ** 2 + vector[1] ** 2))

            cv2.circle(copiedImage, (center_X, center_Y), 7, (255, 255, 255), -1)

            # Vectors.append
            cv2.imshow('Image', copiedImage)

            Time.append(time_counter)
            time_counter += 1 / FPS

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    return [Time, vector_x, vector_y, distance]


def main():
    Time, x, y, distance = processData('0_720_Proccessed_3.mp4')
    # Time, x, y, distance = processData('Magnetic_Field.mp4')
    arr = np.asarray([Time, x, y, distance], dtype=object, )
    pd.DataFrame(arr).to_csv('orbit1.csv')


if __name__ == "__main__":
    main()
