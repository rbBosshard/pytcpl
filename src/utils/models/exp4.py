import numpy as np


def exp4(field):
    def exp4_inverse(y, tp, ga):
        val = 1 - y / tp
        if val == 0:
            val = 1e-6
        return -ga * np.log2(val)

    return {
        "fun": lambda x, tp, ga: tp * (1 - 2 ** (-x / ga)),
        "inv": lambda y, tp, ga, conc=None: exp4_inverse(y, tp, ga),
        "params": ['tp', 'ga', 'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4, 1e6), (0.01, 1e6)),
        "x0": lambda conc=None, resp=None: [100, 10],
        "scale": lambda y, conc, params: y,
    }.get(field)
