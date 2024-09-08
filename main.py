from skimage.transform import (hough_line, hough_line_peaks)
import numpy as np
import cv2
import matplotlib.pyplot as plt


def videoProcess(source):
    output_angle = []

    # Input of the video
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (540, 380), fx=0, fy=0,
                               interpolation=cv2.INTER_CUBIC)
            cropped_img = frame[20:300, 50:325]

            # conversion of BGR to grayscale is necessary to apply this operation
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

            # Blur the image for further process
            img_blur = cv2.GaussianBlur(gray, (3, 3), 0)
            # cv2.imshow('process image', img_blur)
            # cv2.waitKey(0)

            edges = cv2.Canny(image=img_blur, threshold1=150, threshold2=200, apertureSize=3)
            cv2.imshow('Canny Edges', edges)
            # cv2.waitKey(0)

            # Perform Hough Transformation to detect lines
            hspace, angles, distances = hough_line(edges)

            # Find angle
            angle = []
            for _, a, distances in zip(*hough_line_peaks(hspace, angles, distances)):
                angle.append(a)

            # Obtain angle for each line
            angles = [a * 180 / np.pi for a in angle]

            # Compute difference between the two lines
            output_angle.append(np.max(angles) - np.min(angles))

            # define q as the exit button
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    graphData(output_angle)


def graphData(angle):
    length = (1.65866 - 0.65) / 100  # cm -> m
    width = 0.482 / 100  # cm -> m
    height = 1.235 / 1000  # mm -> m
    elastic_modulus = 337843  # in pascal and for Mold Max 20
    moment_inertia = (width * pow(height, 3)) / 12

    force = []
    new_angle = []
    distance = []
    increment = 0
    # Remove outliers
    for i in range(len(angle)):
        if i > 0:
            if angle[i] - angle[i - 1] < 10 and angle[i] != 0 and angle[i] < 90:
                new_angle.append(angle[i])
        else:
            new_angle.append(angle[i])

    # Obtaining the distance
    array_size = len(new_angle)
    distance.append(0)
    if array_size > 1:
        for i in range(1, array_size):
            if new_angle[i] != new_angle[i - 1]:
                increment += 0.01  # in mm
                distance.append(increment)
            else:
                distance.append(increment)

    for a in new_angle:
        rad = a * (np.pi / 180)
        calculated_force = (3 * elastic_modulus * moment_inertia * rad) / pow(length, 3)
        force.append(calculated_force)

    plt.plot(distance, force)
    plt.xlabel("Distance (mm)")
    plt.ylabel("Force (N)")
    plt.show()

    print(new_angle)


def angleDetection(img):
    image = cv2.imread(img)
    # # Display original image
    # cv2.imshow('Original', image)
    # cv2.waitKey(0)

    cropped_img = image[200:600, 100:400]
    # Convert to greyscale
    img_grey = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # Blur image
    img_blur = cv2.GaussianBlur(img_grey, (3, 3), 0)
    cv2.imshow('process image', img_blur)
    cv2.waitKey(0)

    # # Sobel Edge detection
    # sobelx = cv2.Sobel(src=img_grey, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    # sobely = cv2.Sobel(src=img_grey, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    # sobelxy = cv2.Sobel(src=img_grey, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    #
    # # Display Sobel Edge Detection Images
    # cv2.imshow('Sobel X', sobelx)
    # cv2.waitKey(0)
    #
    # cv2.imshow('Sobel Y', sobely)
    # cv2.waitKey(0)
    #
    # cv2.imshow('Sobel XY', sobelxy)
    # cv2.waitKey(0)

    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=150, apertureSize=3)
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)

    # Perform Hough Transformation to detect lines
    hspace, angles, distances = hough_line(edges)

    # Find angle
    angle = []
    for _, a, distances in zip(*hough_line_peaks(hspace, angles, distances)):
        angle.append(a)

    # Obtain angle for each line
    angles = [a * 180 / np.pi for a in angle]

    # Compute difference between the two lines
    angle_difference = np.max(angles) - np.min(angles)
    print(angle_difference)


def main():
    # videoProcess('Adhesion_Mold_Max20_1mm.avi')
    videoProcess('20_1mm_2_2.avi')
    # videoProcess('Friction_Mold_Max_60_1mm.avi')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
