import numpy as np


def get_fit_model(fit_model):
    return {
        'cnst': cnst,
        'exp2': exp2,
        'exp3': exp3,
        'exp4': exp4,
        'exp5': exp5,
        'gnls': gnls,
        'hill': hill,
        'poly1': poly1,
        'poly2': poly2,
        'pow': pow_fn,
    }.get(fit_model)


def get_params(fit_model):
    return dict(cnst=['er'],
                poly1=['a', 'er'],
                poly2=['a', 'b', 'er'],
                pow=['a', 'p', 'er'],
                exp2=['a', 'b', 'er'],
                exp3=['a', 'b', 'p', 'er'],
                exp4=['tp', 'ga', 'er'],
                exp5=['tp', 'ga', 'p', 'er'],
                hill=['tp', 'ga', 'p', 'er'],
                gnls=['tp', 'ga', 'p', 'la', 'q', 'er'],
                ).get(fit_model)


def cnst(ps, x):
    return np.zeros(len(x))


def exp2(ps, x):
    a, b = ps[:2]
    return a * (np.exp(x / b) - 1)


def exp3(ps, x):
    a, b, p = ps[:3]
    return a * (np.exp((x / b) ** p) - 1)


def exp4(ps, x):
    tp, ga = ps[:2]
    return tp * (1 - 2 ** (-x / ga))


def exp5(ps, x):
    tp, ga, p = ps[:3]
    return tp * (1 - 2 ** (-(x / ga) ** p))


def gnls(ps, x):
    tp, ga, p, la, q = ps[:5]
    return tp / ((1 + (ga / x) ** p) * (1 + (x / la) ** q))


def hill(ps, x):
    tp, ga, p = ps[:3]
    return tp / (1 + (ga / x) ** p)


def poly1(ps, x):
    a = ps[:1]
    return a * x


def poly2(ps, x):
    a, b = ps[:2]
    frac = x / b
    return a * (frac + frac ** 2)


def pow_fn(ps, x):
    a, p = ps[:2]
    return a * x ** p
