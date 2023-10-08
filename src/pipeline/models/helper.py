import numpy as np


def pow_space(start, stop, power, num):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


def get_inverse(fit_model, y, x_max, x_min, param, num_points=1000):
    x_range = pow_space(x_min, x_max, 10, num_points)
    y_values = fit_model(x_range, *param)
    x = x_min
    tp = param[0]
    for i in range(num_points - 1):
        # Sandwich y
        test = y_values[i] <= y <= y_values[i + 1] if tp > 0 else y_values[i+1] <= y <= y_values[i]
        if test:
            x = x_range[i]
            break
    return x


def get_er_bounds():
    return -100, 100


def get_er_est(resp):
    rmad = get_mad(resp)
    return np.log(rmad) if rmad > 0 else np.log(1e-4)


def get_mmed_conc(bidirectional, conc, resp):
    max_idx, unique_conc, _ = get_max_index(bidirectional, conc, resp)
    mmed_conc = unique_conc[max_idx]
    return mmed_conc  # mmed = rmds[max_idx]


def get_mmed(bidirectional, conc, resp):
    max_idx, _, rmds = get_max_index(bidirectional, conc, resp)
    mmed = rmds[max_idx]
    return mmed


def get_max_index(bidirectional, conc, resp):
    unique_conc = np.unique(conc)
    # get max response (i.e. max median response for multi-valued responses) and corresponding conc
    if bidirectional:
        rmds = np.array([np.median(abs(resp[conc == c])) for c in unique_conc])
    else:
        rmds = np.array([np.median(resp[conc == c]) for c in unique_conc])
    max_idx = np.argmax(rmds)
    return max_idx, unique_conc, rmds


def get_mad(x):
    """Calculate the median absolute deviation (MAD) of an array"""
    bmad_constant = 1.4826
    return bmad_constant * np.nanmedian(np.abs(x - np.nanmedian(x)))
