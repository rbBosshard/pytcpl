import numpy as np


b = 1e6
a = 0.0
c = 10
e = 5
d = 1e-6
a1 = 0.01
a2 = 1e-4


def get_initial_values(fit_model, conc=[]):
    return {
        'cnst': [0.1, 0.5],
        'poly1': [0.8],
        'poly2': [300, 500],
        'pow': [1.5, 0.8],
        'exp4': [100, 10],
        'exp5': [100, 10, 2],
        'hill': [100, 10, 2],
        'gnls': [100, 10, 2, 10, 0.5],
        'sigmoid': [100, 10, 2, 0.5]
    }.get(fit_model)


def get_bounds(fit_model, conc=[]):
    return {
        'cnst': ((d, b), (-1, 1)),
        'poly1': ((a1, c),),
        'poly2': ((a1, b), (a1, b)),
        'pow': ((a2, 100), (0.1, c)),
        'exp4': ((a2, b), (a1, b)),
        'exp5': ((a2, b), (a1, b), (a1, c)),
        'hill': ((d, b), (d, b), (0.1, c)),
        'gnls': ((d, b), (d, b), (d, c), (a1, b), (d, c)),
        'sigmoid': ((d, b), (a2, b), (d, e), (d, 2)),
    }.get(fit_model)


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
        'sigmoid': sigmoid,
    }.get(fit_model)


def get_inverse_model(fit_model):
    return {
        'exp4': exp4_inverse,
        'exp5': exp5_inverse,
        'gnls': gnls_inverse,
        'hill': hill_inverse,
        'poly1': poly1_inverse,
        'poly2': poly2_inverse,
        'pow': pow_fn_inverse,
        'sigmoid': sigmoid_inverse,
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
                sigmoid=['tp', 'ga', 'p', 'q', 'er']
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


def sigmoid(x, tp, ga, p, q):
    return tp / (1 + (ga / x) ** p) / np.exp(q * x)


# inverse
def poly1_inverse(y, a, conc=[]):
    return y / a


def poly2_inverse(y, a, b, conc=[]):
    return b * (-1 + np.sqrt(1 + 4 * y / a)) / 2


def pow_fn_inverse(y, a, p, conc=[]):

    return (y / a) ** (1 / p)


def exp4_inverse(y, tp, ga, conc=[]):
    val = 1 - y / tp
    if val == 0:
        val = 1e-6
    return -ga * np.log2(val)


def exp5_inverse(y, tp, ga, p, conc=[]):
    val = 1 - y / tp
    if val == 0:
        val = 1e-6
    return ga * (-np.log2(val)) ** (1 / p)


def hill_inverse(y, tp, ga, p, conc=[]):
    val = (tp / y) - 1
    if val == 0:
        val = 1e-6
    return ga / val ** (1 / p)


def gnls_inverse(y, tp, ga, p, la, q, conc=[], x_min_limit=1e-10):
    param = [tp, ga, p, la, q]
    x_min = np.min(conc)
    x_max = np.max(conc)
    x = get_inverse(gnls, y,  x_max, x_min, param)

    # x was not found in visible range
    if x_min == x:
        x_min = x_min_limit
        x_max = np.min(conc)
        x = get_inverse(gnls, y, x_max, x_min, param)
    return x


def sigmoid_inverse(y, tp, ga, p, q, conc=[], x_min_limit=1e-10):
    param = [tp, ga, p, q]
    x_min = np.min(conc)
    x_max = np.max(conc)
    x = get_inverse(sigmoid, y, x_max, x_min, param)

    # x was not found in visible range
    if x_min == x:
        x_min = x_min_limit
        x_max = np.min(conc)
        x = get_inverse(sigmoid, y, x_max, x_min, param)
    return x


def powspace(start, stop, power, num):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


def get_inverse(fit_model, y, x_max, x_min, param, num_points=1000):
    x_range = powspace(x_min, x_max, 10, num_points)
    y_values = fit_model(x_range, *param)
    x = x_min
    for i in range(num_points - 1):
        test = y_values[i] <= y <= y_values[i + 1]
        if test:
            x = x_range[i]
            break
    return x


def scale_for_log_likelihood_at_cutoff(fit_model, cutoff, conc, params):
    if fit_model in ["exp4", "exp5", "hill", "gnls", "sigmoid"]:
        return cutoff
    elif fit_model == "poly1":
        return cutoff / np.max(conc)
    elif fit_model == "poly2":
        return cutoff / (np.max(conc) / params[1] + (np.max(conc) / params[1]) ** 2)
    elif fit_model == "pow":
        return cutoff / (np.max(conc) ** params[1])
    else:
        NotImplementedError()