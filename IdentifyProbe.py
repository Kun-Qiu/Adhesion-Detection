import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline


def main():
    frame = cv2.imread("snip1.png")
    frame = cv2.resize(frame, (600, 600), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh2 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('ogimage', frame)
    cv2.imshow('image', thresh2)
    cv2.waitKey(0)

    medianFiltered = cv2.medianBlur(thresh2, 5)

    # Detecting contour
    contours, hierarchy = cv2.findContours(medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # copiedImage = np.copy(frame)
    # # cv2.drawContours(copiedImage, contours, -1, (255, 255, 0), 3)
    # c = max(contours, key=cv2.contourArea)
    # # Finding the top extreme point in contour
    # extTop = tuple(c[c[:, :, 1].argmin()][0])
    # cv2.drawContours(copiedImage, contours, -1, (255, 255, 0), 3)
    # cv2.circle(copiedImage, extTop, 8, (255, 0, 0), -1)
    # cv2.imshow('extreme',copiedImage)
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()
