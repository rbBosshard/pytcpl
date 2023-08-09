import numpy as np

from .get_inverse import get_inverse


def sigmoid(field):
    def sigmoid_inverse(y, tp, ga, p, q, conc=[], x_min_limit=1e-10):
        param = [tp, ga, p, q]
        x_min = np.min(conc)
        x_max = np.max(conc)
        x = get_inverse(sigmoid('fun'), y, x_max, x_min, param)

        # x was not found in visible range
        if x_min == x:
            x_min = x_min_limit
            x_max = np.min(conc)
            x = get_inverse(sigmoid('fun'), y, x_max, x_min, param)
        return x

    return {
        "fun": lambda x, tp, ga, p, q: tp / (1 + (ga / x) ** p) / np.exp(q * x),
        "inv": lambda y, tp, ga, p, q, conc=None: sigmoid_inverse(y, tp, ga, p, q, conc),
        "params": ['tp', 'ga', 'p', 'q', 'er'],
        "bounds": lambda conc=None, resp=None: ((1e-6, 1e6), (1e-4, 1e6), (1e-6, 5), (1e-6, 2)),
        "x0": lambda conc=None, resp=None: [100, 10, 2, 0.5],
        "scale": lambda y, conc, params: y,
    }.get(field)
