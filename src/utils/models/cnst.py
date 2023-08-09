import numpy as np


def cnst(field):
    return {
        "fun": lambda x, a: np.zeros(len(x)),
        "inv": None,
        "params": ['a', 'er'],
        "bounds": lambda conc=None, resp=None: ((1e-6, 1e6), (-1, 1)),
        "x0": lambda conc=None, resp=None: [0.1, 0.5],
        "scale": None
    }.get(field)
