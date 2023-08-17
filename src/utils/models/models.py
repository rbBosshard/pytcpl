from src.utils.models.cnst import cnst
from src.utils.models.exp4 import exp4
from src.utils.models.exp5 import exp5
from src.utils.models.gnls import gnls
from src.utils.models.hill import hill
from src.utils.models.poly1 import poly1
from src.utils.models.poly2 import poly2
from src.utils.models.pow import pow
from src.utils.models.sigmoid import sigmoid


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
