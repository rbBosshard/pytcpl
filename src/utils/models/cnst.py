import numpy as np
from .helper import get_er_est, get_mmed_conc, get_mmed, get_er_bounds


def cnst(field):
    return {
        "fun": lambda x, a: np.zeros(len(x)),
        "inv": None,
        "params": ['a',
                   'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4, 1e4),
                                                (get_er_bounds())),
        "x0": lambda conc=None, resp=None: [1,
                                            get_er_est(resp)],
        "scale": None
    }.get(field)
