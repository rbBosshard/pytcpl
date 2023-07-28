import numpy as np
from scipy.stats import t, norm


def tcpl_obj(params, conc, resp, fit_model, errfun="dt4"):
    # Optimization objective function is called "cost function" or "loss function"
    # and therefore, we want to minimize them, rather than maximize them,
    # hence the negative log likelihood is formed, wrapped with scipy.optimize.minimize()
    # https://stats.stackexchange.com/questions/260505/why-do-we-use-negative-log-likelihood-to-estimate-parameters-for-ml
    # Objective function is the sum of log-likelihood of response
    # given the model at each concentration scaled by variance (err)
    pred = fit_model(conc, *params[:-1])
    error = resp - pred
    df = 4  # len(conc) - len(params)  # Degrees of freedom
    sigma_squared = np.var(error)
    scale = np.sqrt(sigma_squared)
    # standardized_residuals = error #/ np.sqrt(sigma_squared)
    scale = np.exp(params[-1]) or np.finfo(np.float64).eps
    # log_likelihood = np.sum(t.logpdf(x=resp, df=4, loc=pred, scale=scale))
    # return -log_likelihood  # negative log likelihood scaled by variance


    try:
        log_likelihood = np.sum(t.logpdf(x=resp, df=4, loc=pred, scale=scale))
    except Exception as e:
        print(f"{fit_model} {e}")
        log_likelihood = np.sum(t.logpdf(x=resp, df=4, loc=pred, scale=scale))
    return -log_likelihood  # negative log likelihood scaled by variance


