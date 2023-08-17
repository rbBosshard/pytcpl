from .cnst import cnst
from .exp4 import exp4
from .exp5 import exp5
from .gnls import gnls
from .hill import hill
from .poly1 import poly1
from .poly2 import poly2
from .pow import pow
from .sigmoid import sigmoid


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
