import numpy as np


def poly1(field):
    return {
        "fun": lambda x, a: a * x,
        "inv": lambda y, a, conc=None: y / a,
        "params": ['a', 'er'],
        "bounds": lambda conc=None, resp=None: ((0.01, 10),),
        "x0": lambda conc=None, resp=None: [0.8],
        "scale": lambda y, conc, params: y / np.max(conc),
    }.get(field)
