from scipy.optimize import fsolve
import numpy as np

x = 1000
a = 0.1
c = 10

BOUNDS = {
    'cnst': ((a, x), (-1, 1)),
    'poly1': ((-c, c),),
    'poly2': ((-c, x), (-c, x)),
    'pow': ((-c, 100), (a, c)),
    'exp4': ((a, x), (a, x)),
    'exp5': ((a, x), (a, x), (a, c)),
    'hill': ((a, x), (a, x), (a, c)),
    'gnls': ((a, x), (a, x), (a, c), (a, x), (a, c)),
}

INITIAL_VALUES = {
    'cnst': [a, 0.9],
    'poly1': [0.8],
    'poly2': [230, 340],
    'pow': [1.5, 0.8],
    'exp4': [110, 80],
    'exp5': [100, 50, 1.5],
    'hill': [90, 30, 1.9],
    'gnls': [170, 45, 1.7, 70, 1.7],
}


def get_model(fit_model):
    return {
        'cnst': cnst,
        'exp4': exp4,
        'exp5': exp5,
        'gnls': gnls,
        'hill': hill,
        'poly1': poly1,
        'poly2': poly2,
        'pow': pow_fn,
        'exp4_': exp4_inverted,
        'exp5_': exp5_inverted,
        'gnls_': gnls_inverted,
        'hill_': hill_inverted,
        'poly1_': poly1_inverted,
        'poly2_': poly2_inverted,
        'pow_': pow_fn_inverted,
    }.get(fit_model)


def get_params(fit_model):
    return dict(cnst=['a', 'er'],
                poly1=['a', 'er'],
                poly2=['a', 'b', 'er'],
                pow=['a', 'p', 'er'],
                exp4=['tp', 'ga', 'er'],
                exp5=['tp', 'ga', 'p', 'er'],
                hill=['tp', 'ga', 'p', 'er'],
                gnls=['tp', 'ga', 'p', 'la', 'q', 'er'],
                ).get(fit_model)


def cnst(x, a):
    return np.zeros(len(x))


def poly1(x, a):
    return a * x


def poly2(x, a, b):
    frac = x / b
    return a * (frac + frac ** 2)


def pow_fn(x, a, p):
    return a * x ** p


def exp4(x, tp, ga):
    return tp * (1 - 2 ** (-x / ga))


def exp5(x, tp, ga, p):
    return tp * (1 - 2 ** (-(x / ga) ** p))


def hill(x, tp, ga, p):
    return tp / (1 + (ga / x) ** p)


def gnls(x, tp, ga, p, la, q):
    return tp / ((1 + (ga / x) ** p) * (1 + (x / la) ** q))


# Inverted
def poly1_inverted(y, a):
    return y / a


def poly2_inverted(y, a, b):
    return b * (-1 + np.sqrt(1 + 4 * y / a)) / 2


def pow_fn_inverted(y, a, p):
    return (y / a) ** (1 / p)


def exp4_inverted(y, tp, ga):
    return -ga * np.log2(1 - y / tp)


def exp5_inverted(y, tp, ga, p):
    return ga * (-np.log2(1 - y / tp)) ** (1 / p)


def hill_inverted(y, tp, ga, p):
    return ga / ((tp / y) - 1) ** (1 / p)


def gnls_inverted(y, tp, ga, p, la, q):
    def equation(x):
        return gnls(x, tp, ga, p, la, q) - y
    loc = fsolve(equation, x0=np.array([1.0]))
    return gnls(loc, tp, ga, p, la, q)  # Todo: check correctness, compare with tcpl














