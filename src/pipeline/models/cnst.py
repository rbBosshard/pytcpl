import numpy as np

from .helper import get_er_est, get_er_bounds


def cnst(field):
    """
    Create and return constant model functions and parameters.

    This function generates and returns different components of the constant model, including the forward function,
    inverse function, parameter names, parameter bounds, initial parameter guesses, and scaling function.

    Args:
        field (str): The field corresponding to the desired component of the constant model.

    Returns:
        dict or lambda: Depending on the provided field, either a dictionary containing model information (such as
                        functions, parameter names, bounds, etc.) or a lambda function representing the selected
                        component.

    Note:
    The constant model represents a flat, constant relationship between variables. It includes a forward function to
    calculate constant responses and parameter bounds, as well as initial parameter guesses.

    """
    return {
        "fun": lambda x, a: np.zeros(len(x)),
        "inv": None,
        "params": ['a',
                   'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4, 1e4),
                                                (get_er_bounds())),
        "bounds_bidirectional": lambda conc=None, resp=None: ((1e-4, 1e4),
                                                (get_er_bounds())),
        "x0": lambda bidirectional=True, conc=None, resp=None: [1,
                                            get_er_est(resp)],
        "scale": None
    }.get(field)
