import numpy as np

from ..constants import BMAD_CONSTANT


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


def get_er_bounds():
    return -5, 10


def get_er_est(resp):
    rmad = get_mad(resp)
    return np.log(rmad) if rmad > 0 else np.log(1e-4)


def get_mmed_conc(conc, resp):
    max_idx, unique_conc, _ = get_max_index(conc, resp)
    mmed_conc = unique_conc[max_idx]
    return mmed_conc # mmed = rmds[max_idx]


def get_mmed(conc, resp):
    max_idx, _, rmds= get_max_index(conc, resp)
    mmed = rmds[max_idx]
    return mmed


def get_max_index(conc, resp):
    unique_conc = np.unique(conc)
    # get max response (i.e. max median response for multi-valued responses) and corresponding conc
    rmds = np.array([np.median(resp[conc == c]) for c in unique_conc])
    max_idx = np.argmax(rmds)
    return max_idx, unique_conc, rmds


def get_mad(x):
    """Calculate the median absolute deviation (MAD) of an array"""
    return BMAD_CONSTANT * np.median(np.abs(x - np.median(x)))
