import numpy as np


def poly2(field):
    def poly2_(x, a, b):
        frac = x / b
        return a * (frac + frac ** 2)

    return {
        "fun": lambda x, a, b: poly2_(x, a, b),
        "inv": lambda y, a, b, conc=None: b * (-1 + np.sqrt(1 + 4 * y / a)) / 2,
        "params": ['a', 'b', 'er'],
        "bounds": lambda conc=None, resp=None: ((0.01, 1e6), (0.01, 1e6)),
        "x0": lambda conc=None, resp=None: [300, 500],
        "scale": lambda y, conc, params: y / (np.max(conc) / params[1] + (np.max(conc) / params[1]) ** 2),
    }.get(field)
