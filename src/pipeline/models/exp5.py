import numpy as np

from .helper import get_er_est, get_mmed_conc, get_mmed, get_er_bounds


def exp5(field):
    """
    Create and return exponential decay model (Exp5) functions and parameters.

    This function generates and returns different components of the exponential decay model (Exp5), including the
    forward function, inverse function, parameter names, parameter bounds, initial parameter guesses, and scaling
    function.

    Args:
        field (str): The field corresponding to the desired component of the exponential decay model (Exp5).

    Returns:
        dict or lambda: Depending on the provided field, either a dictionary containing model information (such as
                        functions, parameter names, bounds, etc.) or a lambda function representing the selected
                        component.

    Note:
    The exponential decay model (Exp5) describes a decaying relationship between variables. It includes a forward
    function to calculate responses based on concentrations and model parameters, as well as an inverse function to
    estimate concentrations from responses.

    """
    return {
        "fun": lambda x, tp, ga, p: tp * (1 - 2 ** (-(x / ga) ** p)),
        "inv": lambda y, tp, ga, p, conc=None: ga * (-np.log2(1 - y / tp)) ** (1 / p),
        "params": ['tp',
                   'ga',
                   'p',
                   'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4, 1.2 * abs(get_mmed(False, conc, resp)) or 0.01),
                                                (np.min(conc) / 10, np.max(conc) * np.sqrt(10)),
                                                (0.3, 8),
                                                (get_er_bounds())),
        "bounds_bidirectional": lambda conc=None, resp=None: ((-1.2 * abs(get_mmed(True, conc, resp)), 1.2 * abs(get_mmed(True, conc, resp)) or 0.01),
                                                (np.min(conc) / 10, np.max(conc) * np.sqrt(10)),
                                                (0.3, 8),
                                                (get_er_bounds())),
        "x0": lambda bidirectional=True, conc=None, resp=None: [get_mmed(bidirectional, conc, resp) or 0.01,
                                            get_mmed_conc(bidirectional, conc, resp) / np.sqrt(10),
                                            1.2,
                                            get_er_est(resp)],
        "scale": lambda y, conc, params: y,
    }.get(field)
