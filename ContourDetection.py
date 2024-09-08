import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline, BSpline

# Initial Position Of FootPad
INIT_POS = []
INIT_BASE_POS = []
force = []
time = []

# Displacement of center of contour
displacement = []
delta = []


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        INIT_POS.append(x)
        INIT_POS.append(y)

    if event == cv2.EVENT_RBUTTONDOWN:
        print(x, ' ', y)
        INIT_BASE_POS.append(x)
        INIT_BASE_POS.append(y)


def graphData():
    # Measurement of the beam
    length = (1.65866 - 0.65) / 100  # cm -> m
    width = 0.482 / 100  # cm -> m
    height = 1.235 / 1000  # mm -> m
    elastic_modulus = 337843  # in pascal and for Mold Max 20
    moment_inertia = (width * pow(height, 3)) / 12

    for x in range(len(force)):
        # change delta in radian to force
        force[x] = (3 * elastic_modulus * moment_inertia * force[x]) / pow(length, 3)

    x = np.array(time)
    y = np.array(force)
    xnew = np.linspace(x.min(), x.max(), 200)

    # define spline
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(xnew)

    # create smooth line chart
    plt.plot(xnew, y_smooth)
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.show()


def deflection_angle(pt2, manualPt=None):
    if manualPt is None:
        # Using initial defined points + detection to calculate the deflection angle
        pt1 = [INIT_POS[0], pt2[1]]
        line1 = [pt1[0] - INIT_POS[0], pt1[1] - INIT_POS[1]]
        line2 = [pt2[0] - INIT_POS[0], pt2[1] - INIT_POS[1]]
    else:
        # User defined two points
        pt1 = [manualPt[0], pt2[1]]
        line1 = [pt1[0] - manualPt[0], pt1[1] - manualPt[1]]
        line2 = [pt2[0] - manualPt[0], pt2[1] - manualPt[1]]

    norm_line1 = np.sqrt(pow(line1[0], 2) + pow(line1[1], 2))
    norm_line2 = np.sqrt(pow(line2[0], 2) + pow(line2[1], 2))
    a_dot_b = (line1[0] * line2[0]) + (line1[1] * line2[1])

    cos_theta = np.arccos(a_dot_b / (norm_line1 * norm_line2))
    # print(norm_line1, norm_line2, a_dot_b, cos_theta)
    if INIT_POS[0] - pt2[0] < 0:
        force.append(-cos_theta)
    else:
        force.append(cos_theta)


def main(source):
    time_counter = 0
    automatic = True

    cap = cv2.VideoCapture(source)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (540, 380), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            # Crop image so that contour does not detect other edges
            # cropped_img = frame[0:540, 150:380]

            if time_counter == 0:
                cv2.imshow('image', frame)
                cv2.setMouseCallback('image', click_event)
                cv2.waitKey(0)

            # Automatic for portion when footpad does not detach
            if automatic:
                # Get the initial position of footpad

                # Preprocessing image
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, thresh2 = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
                # cv2.imshow('image', thresh2)
                # cv2.waitKey(0)

                medianFiltered = cv2.medianBlur(thresh2, 5)

                # Detecting contour
                contours, hierarchy = cv2.findContours(medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                copiedImage = np.copy(frame)
                # cv2.drawContours(copiedImage, contours, -1, (255, 255, 0), 3)

                # Finding the contour of interests
                # Should only be one contour
                contourDesired = 0
                for i, c in enumerate(contours):
                    areaContour = cv2.contourArea(c)
                    if 3000 < areaContour:
                        cv2.drawContours(copiedImage, contours, i, (255, 10, 255), 4)
                        contourDesired = c

                # Finding the center of the contour
                # c = max(contours, key=cv2.contourArea)
                M = cv2.moments(contourDesired)
                if M["m00"] != 0:
                    center_X = int(M["m10"] / M["m00"])
                    center_Y = int(M["m01"] / M["m00"])
                else:
                    center_X, center_Y = 0, 0
                cv2.circle(copiedImage, (center_X, center_Y), 7, (255, 255, 255), -1)
                displacement.append(center_X)

                if time_counter == 0:
                    deflection_angle(INIT_BASE_POS)
                    delta.append(INIT_BASE_POS[0])
                else:
                    delta_distance = displacement[1] - displacement[0]
                    # print(displacement[0], displacement[1], delta_distance)
                    delta.append(INIT_BASE_POS[0] + delta_distance)
                    INIT_BASE_POS[0] += delta_distance
                    deflection_angle(INIT_BASE_POS)
                    displacement.pop(0)

                # Append the time
                time.append(time_counter)

                # If m key is pressed change to manual selection
                if cv2.waitKey(1) & 0xFF == ord('m'):
                    automatic = False
                    cv2.destroyAllWindows()

            else:
                INIT_BASE_POS.clear()
                INIT_POS.clear()

                cv2.imshow('image', frame)
                cv2.setMouseCallback('image', click_event)
                cv2.waitKey(0)

                if len(INIT_BASE_POS) != 0 and len(INIT_POS) != 0:
                    tip_Point = [INIT_POS[0], INIT_POS[1]]
                    base_Point = [INIT_BASE_POS[0], INIT_BASE_POS[1]]
                    delta.append(base_Point[0])
                    deflection_angle(tip_Point, base_Point)

                    # Append the time
                    time.append(time_counter)

            cv2.imshow('Image', copiedImage)
            time_counter += 1 / FPS

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Convert the arrays into csv files
    arr = np.asarray([force, delta, time], dtype=object, )
    pd.DataFrame(arr).to_csv('vid2_processed.csv')


if __name__ == "__main__":
    main('vid_processed.mp4')
    graphData()
