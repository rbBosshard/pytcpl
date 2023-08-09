import numpy as np


def pow(field):
    return {
        "fun": lambda x, a, p: a * x ** p,
        "inv": lambda y, a, p, conc=None: (y / a) ** (1 / p),
        "params": ['a', 'p', 'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4, 100), (0.1, 10)),
        "x0": lambda conc=None, resp=None: [1.5, 0.8],
        "scale": lambda y, conc, params: y / (np.max(conc) ** params[1]),
    }.get(field)
