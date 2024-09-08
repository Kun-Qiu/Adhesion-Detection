import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

from deflectionDectection import *
import BodyAngle


def graphData(x, y):
    length = (1.65866 - 0.65) / 100  # cm -> m
    width = 0.482 / 100  # cm -> m
    height = 1.235 / 1000  # mm -> m
    elastic_modulus = 337843  # in pascal and for Mold Max 20
    moment_inertia = (width * pow(height, 3)) / 12

    for i in range(len(x)):
        # change delta in radian to x
        x[i] = (3 * elastic_modulus * moment_inertia * x[i]) / pow(length, 3)

    x_data = np.array(y)
    y_data = np.array(x)
    x_new = np.linspace(x_data.min(), x_data.max(), 200)

    # define spline
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_new)

    # create smooth line chart
    plt.plot(x_new, y_smooth)
    plt.xlabel("y (s)")
    plt.ylabel("x (N)")
    plt.show()


def main():
    angle_force, time_deflection = deflection("pure_chitosan_ph10_processed.mp4")
    print(angle_force)
    print(time_deflection)


if __name__ == "__main__":
    main()
