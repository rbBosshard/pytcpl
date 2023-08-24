import numpy as np

from .helper import get_er_est, get_mmed_conc, get_mmed, get_er_bounds


def hill(field):
    """
    Create and return Hill equation model functions and parameters.

    This function generates and returns different components of the Hill equation model, including the forward function,
    inverse function, parameter names, parameter bounds, initial parameter guesses, and scaling function.

    Args:
        field (str): The field corresponding to the desired component of the Hill equation model.

    Returns:
        dict or lambda: Depending on the provided field, either a dictionary containing model information (such as functions,
                        parameter names, bounds, etc.) or a lambda function representing the selected component.

    Note:
    The Hill equation model is commonly used to describe dose-response relationships. It includes a forward function to
    calculate responses based on concentrations and model parameters, as well as an inverse function to estimate
    concentrations from responses.

    """
    def hill_inverse(y, tp, ga, p):
        epsilon = 1e-4
        try:
            adjusted_y = np.clip(y, epsilon, tp - epsilon)
            denominator = np.clip(tp / adjusted_y - 1, epsilon, np.inf)
            result = ga / (denominator ** (1 / p) + epsilon)
            result = np.clip(result, 0, 1000)  # Clip the result to a reasonable range
            return result
        except (ValueError, ZeroDivisionError, RuntimeWarning):
            raise Exception("Error: Invalid input values")

    return {
        "fun": lambda x, tp, ga, p: tp / (1 + (ga / x) ** p),
        "inv": lambda y, tp, ga, p, conc=None: hill_inverse(y, tp, ga, p),
        "params": ['tp',
                   'ga',
                   'p',
                   'er'],
        "bounds": lambda conc=None, resp=None: ((1e-4, 1.2 * np.max(resp)),
                                                (np.min(conc) / 10, np.max(conc) * np.sqrt(10)),
                                                (0.3, 8),
                                                (get_er_bounds())),
        "x0": lambda conc=None, resp=None: [get_mmed(conc, resp) or 0.01,
                                            get_mmed_conc(conc, resp) / np.sqrt(10),
                                            1.2,
                                            get_er_est(resp)],
        "scale": lambda y, conc, params: y,
    }.get(field)
