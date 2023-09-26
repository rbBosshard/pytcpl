import numpy as np

from .helper import get_er_est, get_mmed, get_er_bounds


def poly2(field):
    """
    Create and return quadratic polynomial equation model functions and parameters.

    This function generates and returns different components of the quadratic polynomial equation model, including the
    forward function, inverse function, parameter names, parameter bounds, initial parameter guesses, and scaling
    function.

    Args:
        field (str): The field corresponding to the desired component of the quadratic polynomial equation model.

    Returns:
        dict or lambda: Depending on the provided field, either a dictionary containing model information (such as
                        functions, parameter names, bounds, etc.) or a lambda function representing the selected
                        component.

    Note:
    The quadratic polynomial equation model represents a second-degree polynomial relationship between variables. It
    includes a forward function to calculate responses based on concentrations and model parameters, as well as an
    inverse function to estimate concentrations from responses.

    """
    def poly2_(x, a, b):
        frac = x / b
        return a * (frac + frac ** 2)

    return {
        "fun": lambda x, a, b: poly2_(x, a, b),
        "inv": lambda y, a, b, conc=None: b * (-1 + np.sqrt(1 + 4 * y / a)) / 2,
        "params": ['a',
                   'b',
                   'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4, 1e8 * abs(get_mmed(False, conc, resp)) or 0.01),
                                                (1e-8 * np.max(conc), 1e8 * np.max(conc)),
                                                (get_er_bounds())),
        "bounds_bidirectional": lambda conc=None, resp=None: ((-1e8 * abs(get_mmed(True, conc, resp)), 1e8 * abs(get_mmed(True, conc, resp)) or 0.01),
                                                (1e-8 * np.max(conc), 1e8 * np.max(conc)),
                                                (get_er_bounds())),
        "x0": lambda bidirectional=True, conc=None, resp=None: [get_mmed(bidirectional, conc, resp) / 2 or 0.01,
                                            np.max(conc),
                                            get_er_est(resp)],
        "scale": lambda y, conc, params: y / (np.max(conc) / params[1] + (np.max(conc) / params[1]) ** 2),
    }.get(field)
