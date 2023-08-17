import numpy as np

from .helper import get_er_est, get_mmed, get_er_bounds


def poly1(field):
    return {
        "fun": lambda x, a: a * x,
        "inv": lambda y, a, conc=None: y / a,
        "params": ['a',
                   'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4, 1e8 * abs(get_mmed(conc, resp))),
                                                (get_er_bounds())),
        "x0": lambda conc=None, resp=None: [get_mmed(conc, resp) or 0.01,
                                            get_er_est(resp)],
        "scale": lambda y, conc, params: y / np.max(conc),
    }.get(field)
