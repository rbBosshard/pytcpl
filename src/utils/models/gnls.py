import numpy as np

from .get_inverse import get_inverse


def gnls(field):
    def gnls_inverse(y, tp, ga, p, la, q, conc=[], x_min_limit=1e-10):
        param = [tp, ga, p, la, q]
        x_min = np.min(conc)
        x_max = np.max(conc)
        x = get_inverse(gnls('fun'), y, x_max, x_min, param)

        # x was not found raw visible range
        if x_min == x:
            x_min = x_min_limit
            x_max = np.min(conc)
            x = get_inverse(gnls('fun'), y, x_max, x_min, param)
        return x

    return {
        "fun": lambda x, tp, ga, p, la, q: tp / ((1 + (ga / x) ** p) * (1 + (x / la) ** q)),
        "inv": lambda y, tp, ga, p, la, q, conc=None: gnls_inverse(y, tp, ga, p, la, q, conc),
        "params": ['tp', 'ga', 'p', 'la', 'q', 'er'],
        "bounds": lambda conc=None, resp=None: ((1e-6, 1e6), (1e-6, 1e6), (1e-6, 10), (0.01, 1e6), (1e-6, 10)),
        "x0": lambda conc=None, resp=None: [100, 10, 2, 10, 0.5],
        "scale": lambda y, conc, params: y,
    }.get(field)
