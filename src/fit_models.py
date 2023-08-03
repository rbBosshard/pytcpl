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
        'expo': expo,
    }.get(fit_model)


def get_params(fit_model):
    return dict(cnst=['a', 'er'],
                poly1=['a', 'er'],
                poly2=['a', 'b', 'er'],
                pow=['a', 'p', 'er'],
                exp2=['a', 'b', 'er'],
                exp3=['a', 'b', 'p', 'er'],
                exp4=['tp', 'ga', 'er'],
                exp5=['tp', 'ga', 'p', 'er'],
                hill=['tp', 'ga', 'p', 'er'],
                gnls=['tp', 'ga', 'p', 'la', 'q', 'er'],
                expo=['A', 'B', 'er'],
                ).get(fit_model)


def expo(x, A, B):
    return A * np.exp(B * x)


def cnst(x, a):
    return np.zeros(len(x))


def poly1(x, a):
    return a * x


def poly2(x, a, b):
    frac = x / b
    return a * (frac + frac ** 2)


def pow_fn(x, a, p):
    return a * x ** p


def exp2(x, a, b):
    print(f"a: {a},  b: {b}")
    return a * (np.exp(x / b) - 1)


def exp3(x, a, b, p):
    return a * (np.exp((x / b) ** p) - 1)


def exp4(x, tp, ga):
    return tp * (1 - 2 ** (-x / ga))


def exp5(x, tp, ga, p):
    return tp * (1 - 2 ** (-(x / ga) ** p))


def gnls(x, tp, ga, p, la, q):
    return tp / ((1 + (ga / x) ** p) * (1 + (x / la) ** q))


def hill(x, tp, ga, p):
    return tp / (1 + (ga / x) ** p)
