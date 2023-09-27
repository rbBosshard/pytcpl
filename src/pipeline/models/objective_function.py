import numpy as np
from scipy.stats import t, norm


def get_negative_log_likelihood(params, conc, resp, fit_model, errfun="dt4"):
    """
    Calculate the negative log-likelihood of a response given a model and its parameters.

    This function calculates the negative log-likelihood of the observed response values given a model and its
    corresponding parameters. The negative log-likelihood is commonly used as a loss function for maximum likelihood
    estimation, where the goal is to minimize this value during optimization.

    Args:
        params (tuple): The model parameters to be used in the fit_model function.
        conc (array-like): Array of concentration values at which the model's predictions will be evaluated.
        resp (array-like): Array of observed response values corresponding to the concentration values.
        fit_model (callable): The model function that generates predictions based on the input parameters.
        errfun (str, optional): The name of the error function used to scale the likelihood. Default is "dt4".

    Returns:
        float: The negative log-likelihood value.

    Note:
    The negative log-likelihood quantifies how well the given model with its parameters explains the observed data.
    A lower negative log-likelihood indicates a better fit of the model to the data.
    Maximum likelihood estimation = Minimizing negative log-likelihood
    Optimization objective function is called "loss/cost function" and we want to minimize the loss/cost
    https://stats.stackexchange.com/questions/260505/why-do-we-use-negative-log-likelihood-to-estimate-parameters-for-ml
    """
    pred = fit_model(conc, *params[:-1])

    scale = np.exp(params[-1])
    df = 4  # len(conc) - len(params)  # Degrees of freedom
    # error = resp - pred
    # sigma_squared = np.var(error)
    # scale = np.sqrt(sigma_squared)
    if errfun == "dt4":
        log_likelihood = np.sum(t.logpdf(x=resp, df=df, loc=pred, scale=scale) - np.log(scale))
    else:  # errfun == "dnorm":
        log_likelihood = np.sum(norm.logpdf(x=resp, loc=pred, scale=scale) - np.log(scale))

    return -log_likelihood
