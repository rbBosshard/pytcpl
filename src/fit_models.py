import numpy as np

b = 1e6
a = 0.0
c = 10
e = 5
d = 1e-6
a1 = 0.01
a2 = 1e-4


def get_model(fit_model):
    return {
        'cnst': cnst,
        'exp4': exp4,
        'exp5': exp5,
        'gnls': gnls,
        'hill': hill,
        'poly1': poly1,
        'poly2': poly2,
        'pow': pow,
        'sigmoid': sigmoid,
    }.get(fit_model)


def cnst(field):
    return {
        "fun": lambda x, a: np.zeros(len(x)),
        "inv": None,
        "params": ['a', 'er'],
        "bounds": lambda conc=None, resp=None: ((d, b), (-1, 1)),
        "x0": lambda conc=None, resp=None: [0.1, 0.5],
        "scale": None
    }.get(field)


def pow(field):
    return {
        "fun": lambda x, a, p: a * x ** p,
        "inv": lambda y, a, p, conc=None: (y / a) ** (1 / p),
        "params": ['a', 'p', 'er'],
        "bounds": lambda conc=None, resp=None: ((a2, 100), (0.1, c)),
        "x0": lambda conc=None, resp=None: [1.5, 0.8],
        "scale": lambda y, conc, params: y / (np.max(conc) ** params[1]),
    }.get(field)


def poly1(field):
    return {
        "fun": lambda x, a: a * x,
        "inv": lambda y, a, conc=None: y / a,
        "params": ['a', 'er'],
        "bounds": lambda conc=None, resp=None: ((a1, c),),
        "x0": lambda conc=None, resp=None: [0.8],
        "scale": lambda y, conc, params: y / np.max(conc),
    }.get(field)


def poly2(field):
    def poly2_(x, a, b):
        frac = x / b
        return a * (frac + frac ** 2)

    return {
        "fun": lambda x, a, b: poly2_(x, a, b),
        "inv": lambda y, a, b, conc=None: b * (-1 + np.sqrt(1 + 4 * y / a)) / 2,
        "params": ['a', 'b', 'er'],
        "bounds": lambda conc=None, resp=None: ((a1, b), (a1, b)),
        "x0": lambda conc=None, resp=None: [300, 500],
        "scale": lambda y, conc, params: y / (np.max(conc) / params[1] + (np.max(conc) / params[1]) ** 2),
    }.get(field)


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
        "bounds": lambda conc=None, resp=None: ((a2, b), (a1, b)),
        "x0": lambda conc=None, resp=None: [100, 10],
        "scale": lambda y, conc, params: y,
    }.get(field)


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
        "bounds": lambda conc=None, resp=None: ((a2, b), (a1, b), (a1, c)),
        "x0": lambda conc=None, resp=None: [100, 10, 2],
        "scale": lambda y, conc, params: y,
    }.get(field)


def hill(field):
    def hill_inverse(y, tp, ga, p):
        val = (tp / y) - 1
        if val == 0:
            val = 1e-6
        return ga / val ** (1 / p)

    return {
        "fun": lambda x, tp, ga, p: tp / (1 + (ga / x) ** p),
        "inv": lambda y, tp, ga, p, conc=None: hill_inverse(y, tp, ga, p),
        "params": ['tp', 'ga', 'p', 'er'],
        "bounds": lambda conc=None, resp=None: ((d, b), (d, b), (0.1, c)),
        "x0": lambda conc=None, resp=None: [100, 10, 2],
        "scale": lambda y, conc, params: y,
    }.get(field)


def gnls(field):
    def gnls_inverse(y, tp, ga, p, la, q, conc=[], x_min_limit=1e-10):
        param = [tp, ga, p, la, q]
        x_min = np.min(conc)
        x_max = np.max(conc)
        x = get_inverse(gnls('fun'), y, x_max, x_min, param)

        # x was not found in visible range
        if x_min == x:
            x_min = x_min_limit
            x_max = np.min(conc)
            x = get_inverse(gnls('fun'), y, x_max, x_min, param)
        return x

    return {
        "fun": lambda x, tp, ga, p, la, q: tp / ((1 + (ga / x) ** p) * (1 + (x / la) ** q)),
        "inv": lambda y, tp, ga, p, la, q, conc=None: gnls_inverse(y, tp, ga, p, la, q, conc),
        "params": ['tp', 'ga', 'p', 'la', 'q', 'er'],
        "bounds": lambda conc=None, resp=None: ((d, b), (d, b), (d, c), (a1, b), (d, c)),
        "x0": lambda conc=None, resp=None: [100, 10, 2, 10, 0.5],
        "scale": lambda y, conc, params: y,
    }.get(field)


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
        "bounds": lambda conc=None, resp=None: ((d, b), (a2, b), (d, e), (d, 2)),
        "x0": lambda conc=None, resp=None: [100, 10, 2, 0.5],
        "scale": lambda y, conc, params: y,
    }.get(field)


def pow_space(start, stop, power, num):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


def get_inverse(fit_model, y, x_max, x_min, param, num_points=1000):
    x_range = pow_space(x_min, x_max, 10, num_points)
    y_values = fit_model(x_range, *param)
    x = x_min
    for i in range(num_points - 1):
        test = y_values[i] <= y <= y_values[i + 1]
        if test:
            x = x_range[i]
            break
    return x
