import numpy as np
from scipy.stats import t


def get_negative_log_likelihood(params, conc, resp, fit_model, errfun="dt4"):
    # Maximum likelihood estimation = Minimizing negative log-likelihood
    # Optimization objective function is called "loss/cost function" and we want to minimize the loss/cost
    # https://stats.stackexchange.com/questions/260505/why-do-we-use-negative-log-likelihood-to-estimate-parameters-for-ml
    # Returns negative sum of log-likelihood of response given the model at each conc scaled by variance (err)
    pred = fit_model(conc, *params[:-1])
    scale = np.exp(params[-1])
    df = 4  # len(conc) - len(params)  # Degrees of freedom
    # error = resp - pred
    # sigma_squared = np.var(error)
    # scale = np.sqrt(sigma_squared)
    log_likelihood = np.sum(t.logpdf(x=resp, df=df, loc=pred, scale=scale))
    return -log_likelihood
