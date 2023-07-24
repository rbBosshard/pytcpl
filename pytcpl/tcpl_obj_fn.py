import numpy as np
from scipy.stats import t, norm


def tcpl_obj(ps, conc, resp, fit_model, fit_strategy, errfun="dt4"):
    # Optimization objective function is called "cost function" or "loss function"
    # and therefore, we want to minimize them, rather than maximize them,
    # hence the negative log likelihood is formed, wrapped with scipy.optimize.minimize()
    # https://stats.stackexchange.com/questions/260505/why-do-we-use-negative-log-likelihood-to-estimate-parameters-for-ml

    # Objective function is the sum of log-likelihood of response
    # given the model at each concentration scaled by variance (err)
    if fit_strategy == "mle":
        pred = fit_model(conc, *ps[:-1])
    else:
        pred = fit_model(conc, *ps)
    # ps = parameter vector, get model values for each conc,
    err = np.exp(ps[-1]) or np.finfo(np.float64).eps  # last parameter is the log of the error/variance
    # residuals = (resp - pred) / err
    if errfun == "dt4":
        # degree of freedom parameter = 4, for Student’s t probability density function
        # t.logpdf(x, df, loc, scale) is identically equivalent to t.logpdf(y, df) / scale with y = (x - loc) / scale.
        # ll = np.sum(t.logpdf(x=resp, df=4, loc=pred, scale=err) - np.log(err))
        ll = np.sum(t.logpdf(x=resp, df=4, loc=pred, scale=err))
    else:  # errfun == "dnorm":
        ll = np.sum(norm.logpdf(x=resp, loc=pred, scale=err) - np.log(err))

    neg_ll = -1 * ll
    return neg_ll  # negative log likelihood scaled by variance

