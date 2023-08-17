import numpy as np

from .helper import get_er_est, get_mmed, get_er_bounds


def poly2(field):
    def poly2_(x, a, b):
        frac = x / b
        return a * (frac + frac ** 2)

    return {
        "fun": lambda x, a, b: poly2_(x, a, b),
        "inv": lambda y, a, b, conc=None: b * (-1 + np.sqrt(1 + 4 * y / a)) / 2,
        "params": ['a',
                   'b',
                   'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4, 1e8 * abs(get_mmed(conc, resp))),
                                                (1e-8 * np.max(conc), 1e8 * np.max(conc)),
                                                (get_er_bounds())),
        "x0": lambda conc=None, resp=None: [get_mmed(conc, resp)/2 or 0.01,
                                            np.max(conc),
                                            get_er_est(resp)],
        "scale": lambda y, conc, params: y / (np.max(conc) / params[1] + (np.max(conc) / params[1]) ** 2),
    }.get(field)
