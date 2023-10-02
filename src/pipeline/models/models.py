from .cnst import cnst
from .exp2 import exp2
from .exp3 import exp3
from .exp4 import exp4
from .exp5 import exp5
from .gnls import gnls
from .hill import hill
from .poly1 import poly1
from .poly2 import poly2
from .pow import pow
from .gnls2 import gnls2


def get_model(fit_model):
    """
    Retrieve the model function based on the provided fit model name.

    This function takes a fit model name as input and returns the corresponding model function that can be used for
    curve fitting. The available fit models include constant, exponential decay (Exp4 and Exp5), gainloss (GNLS),
    Hill, polynomial (poly1 and poly2), power-law (pow), and gnls2 models.

    Args:
        fit_model (str): The name of the fit model for which to retrieve the corresponding model function.

    Returns:
        dict or lambda: Depending on the provided fit_model, either a dictionary containing model information or a
                        lambda function representing the selected model.

    Note:
    The function is designed to provide access to different curve fitting model functions based on the given model
    name.

    """
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
        'pow': pow,
        'gnls2': gnls2,
    }.get(fit_model)
