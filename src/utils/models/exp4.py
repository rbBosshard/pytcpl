import numpy as np
from .helper import get_er_est, get_mmed_conc, get_mmed, get_er_bounds


def exp4(field):
    def exp4_inverse(y, tp, ga):
        epsilon = 1e-4
        try:
            adjusted_y = np.clip(y, epsilon, tp - epsilon)
            exponent = np.clip(-np.log2(np.maximum(1 - adjusted_y / tp, epsilon)) + epsilon, 0, np.inf)
            result = ga * exponent
            return result
        except (ValueError, RuntimeWarning):
            raise Exception("Error: Invalid input values")

    return {
        "fun": lambda x, tp, ga: tp * (1 - 2 ** (-x / ga)),
        "inv": lambda y, tp, ga, conc=None: exp4_inverse(y, tp, ga),
        "params": ['tp',
                   'ga',
                   'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4, 1.2 * abs(get_mmed(conc, resp))),
                                                (np.min(conc) / 10, np.max(conc) * np.sqrt(10)),
                                                (get_er_bounds())),
        "x0": lambda conc=None, resp=None: [get_mmed(conc, resp) or 0.01,
                                            get_mmed_conc(conc, resp) / np.sqrt(10),
                                            get_er_est(resp)],
        "scale": lambda y, conc, params: y,
    }.get(field)
