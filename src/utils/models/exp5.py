import numpy as np


def exp5(field):
    def exp5_inverse(y, tp, ga, p):
        val = 1 - y / tp
        if val == 0:
            val = 1e-6
        return ga * (-np.log2(val)) ** (1 / p)

    return {
        "fun": lambda x, tp, ga, p: tp * (1 - 2 ** (-(x / ga) ** p)),
        "inv": lambda y, tp, ga, p, conc=None: exp5_inverse(y, tp, ga, p),
        "params": ['tp', 'ga', 'p', 'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4, 1e6), (0.01, 1e6), (0.01, 10)),
        "x0": lambda conc=None, resp=None: [100, 10, 2],
        "scale": lambda y, conc, params: y,
    }.get(field)
