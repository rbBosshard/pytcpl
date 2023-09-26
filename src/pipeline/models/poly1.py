import numpy as np

from .helper import get_er_est, get_mmed, get_er_bounds


def poly1(field):
    """
    Create and return linear polynomial equation model functions and parameters.

    This function generates and returns different components of the linear polynomial equation model, including the
    forward function, inverse function, parameter names, parameter bounds, initial parameter guesses, and scaling
    function.

    Args:
        field (str): The field corresponding to the desired component of the linear polynomial equation model.

    Returns:
        dict or lambda: Depending on the provided field, either a dictionary containing model information (such as
                        functions, parameter names, bounds, etc.) or a lambda function representing the selected
                        component.

    Note:
    The linear polynomial equation model represents a first-degree polynomial relationship between variables. It
    includes a forward function to calculate responses based on concentrations and model parameters, as well as an
    inverse function to estimate concentrations from responses.

    """
    return {
        "fun": lambda x, a: a * x,
        "inv": lambda y, a, conc=None: y / a,
        "params": ['a',
                   'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4, 1e8 * abs(get_mmed(False, conc, resp)) or 0.01),
                                                (get_er_bounds())),
        "bounds_bidirectional": lambda conc=None, resp=None: ((-1e8 * abs(get_mmed(True, conc, resp)), 1e8 * abs(get_mmed(True, conc, resp)) or 0.01),
                                                (get_er_bounds())),
        "x0": lambda bidirectional=True, conc=None, resp=None: [get_mmed(bidirectional, conc, resp) or 0.01,
                                            get_er_est(resp)],
        "scale": lambda y, conc, params: y / np.max(conc),
    }.get(field)
