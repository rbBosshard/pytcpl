import numpy as np
from .helper import get_er_est, get_mmed_conc, get_mmed, get_er_bounds


def pow(field):
    return {
        "fun": lambda x, a, p: a * x ** p,
        "inv": lambda y, a, p, conc=None: (y / a) ** (1 / p),
        "params": ['a',
                   'p',
                   'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4 * abs(get_mmed(conc, resp)), 1e8 * abs(get_mmed(conc, resp))),
                                                (0.3, 8),
                                                (get_er_bounds())),
        "x0": lambda conc=None, resp=None: [get_mmed(conc, resp) or 0.01,
                                            1.5,
                                            get_er_est(resp)],
        "scale": lambda y, conc, params: y / (np.max(conc) ** params[1]),
    }.get(field)
