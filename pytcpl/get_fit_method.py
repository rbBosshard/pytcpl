import numpy as np


def get_fit_method(fit_method):
    return {
        'cnst': lambda ps, x: np.zeros(len(x)),  # ignores ps
        'exp2': lambda ps, x: ps[0] * (np.exp(x / ps[1]) - 1),  # a = ps[0], b = ps[1]
        'exp3': lambda ps, x: ps[0] * (np.exp((x / ps[1]) ** ps[2]) - 1),  # a = ps[0], b = ps[1], p = ps[2]
        'exp4': lambda ps, x: ps[0] * (1 - 2 ** (-x / ps[1])),  # tp = ps[0], ga = ps[1]
        'exp5': lambda ps, x: ps[0] * (1 - 2 ** (-(x / ps[1]) ** ps[2])),  # tp = ps[0], ga = ps[1], p = ps[2]
        'gnls_': lambda ps, x: ps[0] * (1 / (1 + (ps[1] / x) ** ps[2])) * (1 / (1 + (x / ps[3]) ** ps[4])),  # tp = ps[0], ga = ps[1], p = ps[2], la = ps[3], q = ps[4]
        'gnls': lambda ps, x: ps[0] * (1 / (1 + 10 ** ((ps[1] - x) * ps[2]))) * (1 / (1 + 10 ** ((x - ps[3]) * ps[4]))),  # tp = ps[0], ga = ps[1], p = ps[2], la = ps[3], q = ps[4]
        'hill_': lambda ps, x: ps[0] / (1 + (ps[1] / x) ** ps[2]),  # tp = ps[0], ga = ps[1], p = ps[2]
        'hill': lambda ps, x: ps[0] / (1 + 10 ** (ps[2] * (ps[1] - x))),  # tp = ps[0], ga = ps[1], p = ps[2]
        'poly1': lambda ps, x: ps[0] * x,  # a = ps[0]
        'poly2': lambda ps, x: ps[0] * (x / ps[1] + (x / ps[1]) ** 2),  # a = ps[0], b = ps[1]
        'pow': lambda ps, x: ps[0] * x ** ps[1]  # a = ps[0], p = ps[1]
    }.get(fit_method)
