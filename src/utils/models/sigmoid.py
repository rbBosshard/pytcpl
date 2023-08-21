import numpy as np

from .helper import get_er_est, get_mmed_conc, get_mmed, get_er_bounds
from .helper import get_inverse


def sigmoid(field):
    """
    Create and return sigmoid equation model functions and parameters.

    This function generates and returns different components of the sigmoid equation model, including the forward
    function, inverse function, parameter names, parameter bounds, initial parameter guesses, and scaling function.

    Args:
        field (str): The field corresponding to the desired component of the sigmoid equation model.

    Returns:
        dict or lambda: Depending on the provided field, either a dictionary containing model information (such as
                        functions, parameter names, bounds, etc.) or a lambda function representing the selected
                        component.

    Note:
    The sigmoid equation model is commonly used to describe dose-response relationships. It includes a forward function
    to calculate responses based on concentrations and model parameters, as well as an inverse function to estimate
    concentrations from responses.

    """
    def sigmoid_inverse(y, tp, ga, p, q, conc=None, x_min_limit=1e-10):
        param = [tp, ga, p, q]
        x_min = np.min(conc)
        x_max = np.max(conc)
        x = get_inverse(sigmoid('fun'), y, x_max, x_min, param)

        # x was not found raw visible range
        if x_min == x:
            x_min = x_min_limit
            x_max = np.min(conc)
            x = get_inverse(sigmoid('fun'), y, x_max, x_min, param)
        return x

    return {
        "fun": lambda x, tp, ga, p, q: tp / (1 + (ga / x) ** p) / np.exp(q * x),
        "inv": lambda y, tp, ga, p, q, conc=None: sigmoid_inverse(y, tp, ga, p, q, conc),
        "params": ['tp',
                   'ga',
                   'p',
                   'q',
                   'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4, 1.2 * np.max(resp)),
                                                (np.min(conc) / 10, np.max(conc) * np.sqrt(10)),
                                                (1e-4, 5),
                                                (1e-4, 2),
                                                (get_er_bounds())),
        "x0": lambda conc=None, resp=None: [get_mmed(conc, resp) or 0.01,
                                            get_mmed_conc(conc, resp) / np.sqrt(10),
                                            1.2,
                                            0.5,
                                            get_er_est(resp)],
        "scale": lambda y, conc, params: y,
    }.get(field)
