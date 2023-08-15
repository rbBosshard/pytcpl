import numpy as np
from .helper import get_er_est, get_mmed_conc, get_mmed, get_er_bounds


def exp5(field):
    def exp5_inverse(y, tp, ga, p):
        epsilon = 1e-4
        try:
            adjusted_y = np.clip(y, epsilon, tp - epsilon)
            result = ga * (-np.log2(1 - adjusted_y / tp + epsilon)) ** (1 / p)
            result = np.clip(result, 0, 1000)
            return result
        except (ValueError, ZeroDivisionError, RuntimeWarning):
            print("Error: Invalid input values")
            return None
    return {
        "fun": lambda x, tp, ga, p: tp * (1 - 2 ** (-(x / ga) ** p)),
        "inv": lambda y, tp, ga, p, conc=None: exp5_inverse(y, tp, ga, p),
        "params": ['tp',
                   'ga',
                   'p',
                   'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4, 1.2 * abs(get_mmed(conc, resp))),
                                                (np.min(conc) / 10, np.max(conc) * np.sqrt(10)),
                                                (0.3, 8),
                                                (get_er_bounds())),
        "x0": lambda conc=None, resp=None: [get_mmed(conc, resp) or 0.01,
                                            get_mmed_conc(conc, resp) / np.sqrt(10),
                                            1.2,
                                            get_er_est(resp)],
        "scale": lambda y, conc, params: y,
    }.get(field)
