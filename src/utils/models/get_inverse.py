import numpy as np


def pow_space(start, stop, power, num):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


def get_inverse(fit_model, y, x_max, x_min, param, num_points=1000):
    x_range = pow_space(x_min, x_max, 10, num_points)
    y_values = fit_model(x_range, *param)
    x = x_min
    for i in range(num_points - 1):
        test = y_values[i] <= y <= y_values[i + 1]
        if test:
            x = x_range[i]
            break
    return x
