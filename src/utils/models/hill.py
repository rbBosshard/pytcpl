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
        "bounds": lambda conc=None, resp=None: ((1e-6, 1e6), (1e-6, 1e6), (0.1, 10)),
        "x0": lambda conc=None, resp=None: [100, 10, 2],
        "scale": lambda y, conc, params: y,
    }.get(field)
