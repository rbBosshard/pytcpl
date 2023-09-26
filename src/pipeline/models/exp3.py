import numpy as np

from .helper import get_er_est, get_mmed_conc, get_mmed, get_er_bounds


def exp3(field):
    """
    Create and return exponential decay model (Exp) functions and parameters.

    This function generates and returns different components of the exponential decay model (Exp4), including the
    forward function, inverse function, parameter names, parameter bounds, initial parameter guesses, and scaling
    function.

    Args:
        field (str): The field corresponding to the desired component of the exponential decay model (Exp4).

    Returns:
        dict or lambda: Depending on the provided field, either a dictionary containing model information (such as
                        functions, parameter names, bounds, etc.) or a lambda function representing the selected
                        component.

    Note:
    The exponential decay model (Exp4) describes a decaying relationship between variables. It includes a forward
    function to calculate responses based on concentrations and model parameters, as well as an inverse function to
    estimate concentrations from responses.

    """
    # issue: np.exp((x / b) ** p overflows (with defined bounds),
    return {
        "fun": lambda x, a, b, p: a * (np.exp((x / b) ** p) - 1),
        "inv": lambda y, a, b, p, conc=None: b * (np.log(y / a + 1)) ** (1 / p),
        "params": ['a',
                   'b',
                   'p',
                   'er'],
        "bounds": lambda conc=None, resp=None: ((1e-8 * abs(get_mmed(False, conc, resp)), 1e8 * abs(get_mmed(False, conc, resp)) or 0.01),
                                                (1e-2 * np.max(conc), 1e8 * np.max(conc)),
                                                (0.3, 8),
                                                (get_er_bounds())),
        "bounds_bidirectional": lambda conc=None, resp=None: ((-1e8 * abs(get_mmed(True, conc, resp)), 1e8 * abs(get_mmed(True, conc, resp)) or 0.01),
                                                (1e-2 * np.max(conc), 1e8 * np.max(conc)),
                                                (0.3, 8),
                                                (get_er_bounds())),
        "x0": lambda bidirectional=True, conc=None, resp=None: [get_mmed(bidirectional, conc, resp) or 0.01,
                                            np.max(conc),
                                            1.2,
                                            get_er_est(resp)],
        "scale": lambda y, conc, params: y / (np.exp((np.max(conc) / params[1]) ** params[2]) - 1),
    }.get(field)
