from .models.cnst import cnst
from .models.exp4 import exp4
from .models.exp5 import exp5
from .models.gnls import gnls
from .models.hill import hill
from .models.poly1 import poly1
from .models.poly2 import poly2
from .models.pow import pow
from .models.sigmoid import sigmoid


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


