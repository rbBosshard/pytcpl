import numpy as np
import scipy.optimize as optimize
from scipy.stats import chi2

from pytcpl.acy import tcpl_obj, acy

from pytcpl.get_params import get_params
from pytcpl.get_fit_method import get_fit_method


def bmd_bounds(fit_method, bmr, pars, conc, resp, onesidedp=0.05, bmd=None, which_bound="lower"):
    """
    BMD Bounds

    Uses maximum likelihood method to tune the upper and lower bounds on the BMD (BMDU, BMDL).

    Takes in concentration response fit details and outputs a bmdu or bmdl, as desired.
    If bmd is not finite, returns NA. If the objective function doesn't change sign or
    the root finding otherwise fails, it returns NA. These failures are not uncommon
    since some curves just don't reach the desired confidence level.

    Parameters:
    -----------
    fit_method : str
        Fit method: "exp2", "exp3", "exp4", "exp5", "hill", "gnls", "poly1", "poly2", or "pow".
    bmr : float
        Benchmark response.
    pars : dict
        Named vector of model parameters: a,b,tp,ga,p,la,q,er output by httrfit, and in that order.
    conc : array-like
        Vector of concentrations (NOT in log units).
    resp : array-like
        Vector of responses corresponding to given concentrations.
    onesidedp : float
        The one-sided p-value. Default of .05 corresponds to 5 percentile BMDL, 95 percentile BMDU, and 90 percent CI.
    bmd : float, optional
        Can optionally input the bmd when already known to avoid unnecessary calculation.
    which.bound : str, optional
        Returns BMDU if which.bound = "upper"; returns BMDL if which.bound = "lower".

    Returns:
    --------
    float or None
        Returns either the BMDU or BMDL.

    Examples:
    ---------
    conc = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    resp = [0.1, -0.1, 0, 1.1, 1.9, 2, 2.1, 1.9]
    pars = {'tp': 1.973356, 'ga': 0.9401224, 'p': 3.589397, 'er': -2.698579}
    bmdbounds(fit_method="hill", bmr=0.5, pars=pars, conc=conc, resp=resp)
    bmdbounds(fit_method="hill", bmr=0.5, pars=pars, conc=conc, resp=resp, which.bound="upper")
    """

    # calculate bmd, if necessary
    if bmd is None:
        bmd = acy(bmr, pars, fit_method=fit_method)
    if not np.isfinite(bmd):
        return np.nan

    params = [pars[key] for key in get_params(fit_method)]

    # Todo: recheck if right method is applied (also in R package), i.e. log variant
    if fit_method in ["hill", "gnls"]:
        fit_method += "_"

    # negated max negative loglikelihood. Todo: recheck if everything is correct like this
    maxloglik = -tcpl_obj(ps=params, conc=conc, resp=resp, fit_method=get_fit_method(fit_method))

    # search for bounds to ensure sign change
    bmdrange = None
    if which_bound == "lower":
        xs = 10 ** np.linspace(-5, np.log10(bmd), num=100)
        ys = np.array([bmd_obj(x, fit_method=fit_method, bmr=bmr, conc=conc, resp=resp, ps=pars, mll=maxloglik,
                               onesp=onesidedp, partype=2) for x in xs])
        if not np.any(ys >= 0) or not np.any(ys < 0):
            return np.nan
        bmdrange = np.array([np.max(xs[ys >= 0]), bmd])

    elif which_bound == "upper":
        if fit_method == "gnls":
            toploc = acy(bmr, pars, fit_method="gnls", returntoploc=True)
            xs = 10 ** np.linspace(np.log10(bmd), np.log10(toploc), num=100)
        else:
            xs = 10 ** np.linspace(np.log10(bmd), 5, num=100)
        ys = np.array([bmd_obj(x, fit_method=fit_method, bmr=bmr, conc=conc, resp=resp, ps=pars, mll=maxloglik,
                               onesp=onesidedp, partype=2) for x in xs])
        if not np.any(ys >= 0) or not np.any(ys < 0):
            return np.nan
        bmdrange = np.array([bmd, np.min(xs[ys >= 0])])

    try:
        # use type 2 param. only
        return optimize.root_scalar(bmd_obj, bracket=bmdrange,
                                    args=(fit_method, bmr, conc, resp, pars, maxloglik, onesidedp, 2)).root
    except ValueError:
        return np.nan


def bmd_obj(bmd, fit_method, bmr, conc, resp, ps, mll, onesp, partype=2):
    # implements the BMD substitutions in Appendix A of the Technical Report.
    # Changes one of the existing parameters to an explicit bmd parameter through
    # the magic of algebra.
    if fit_method == "exp2":
        if partype == 1:
            ps["a"] = bmr / (np.exp(bmd / ps["b"]) - 1)
        elif partype == 2:
            ps["b"] = bmd / (np.log(bmr / ps["a"] + 1))
        elif partype == 3:
            ps["b"] = bmd / (np.log(bmr / ps["a"] + 1))
    elif fit_method == "exp3":
        if partype == 1:
            ps["a"] = bmr / (np.exp((bmd / ps["b"]) ** ps["p"]) - 1)
        elif partype == 2:
            ps["b"] = bmd / (np.log(bmr / ps["a"] + 1)) ** (1 / ps["p"])
        elif partype == 3:
            ps["p"] = np.log(np.log(bmr / ps["a"] + 1)) / np.log(bmd / ps["b"])
    elif fit_method == "exp4":
        if partype == 1:
            ps["tp"] = bmr / (1 - 2 ** (-bmd / ps["ga"]))
        elif partype == 2:
            ps["ga"] = bmd / (-np.log2(1 - bmr / ps["tp"]))
        elif partype == 3:
            ps["ga"] = bmd / (-np.log2(1 - bmr / ps["tp"]))
    elif fit_method == "exp5":
        if partype == 1:
            ps["tp"] = bmr / (1 - 2 ** (-(bmd / ps["ga"]) ** ps["p"]))
        elif partype == 2:
            ps["ga"] = bmd / ((-np.log2(1 - bmr / ps["tp"])) ** (1 / ps["p"]))
        elif partype == 3:
            ps["p"] = np.log(-np.log2(1 - bmr / ps["tp"])) / np.log(bmd / ps["ga"])
    elif fit_method == "hill_":
        if partype == 1:
            ps["tp"] = bmr * (1 + (ps["ga"] / bmd) ** ps["p"])
        elif partype == 2:
            ps["ga"] = bmd * (ps["tp"] / bmr - 1) ** (1 / ps["p"])
        elif partype == 3:
            ps["p"] = np.log(ps["tp"] / bmr - 1) / np.log(ps["ga"] / bmd)
    if fit_method == "gnls_":
        if partype == 1:
            ps["tp"] = bmr * ((1 + (ps["ga"] / bmd) ** ps["p"]) * (1 + (bmd / ps["la"]) ** ps["q"]))
        elif partype == 2:
            ps["ga"] = bmd * ((ps["tp"] / (bmr * (1 + (bmd / ps["la"]) ** ps["q"]))) - 1) ** (1 / ps["p"])
        elif partype == 3:
            ps["p"] = np.log(ps["tp"] / (bmr * (1 + (bmd / ps["la"]) ** ps["q"]))) - 1 / np.log(ps["ga"] / bmd)
    elif fit_method == "poly1":
        if partype == 1:
            ps["a"] = bmr / bmd
        elif partype == 2:
            ps["a"] = bmr / bmd
        elif partype == 3:
            ps["a"] = bmr / bmd
    elif fit_method == "poly2":
        if partype == 1:
            ps["a"] = bmr / (bmd / ps["b"] + (bmd / ps["b"]) ** 2)
        elif partype == 2:
            ps["b"] = 2 * bmd / (np.sqrt(1 + 4 * bmr / ps["a"]) - 1)
        elif partype == 3:
            ps["b"] = 2 * bmd / (np.sqrt(1 + 4 * bmr / ps["a"]) - 1)
    elif fit_method == "pow":
        if partype == 1:
            ps["a"] = bmr / (bmd ** ps["p"])
        elif partype == 2:
            ps["a"] = bmr / (bmd ** ps["p"])
        elif partype == 3:
            ps["p"] = np.log(bmr / ps["a"]) / np.log(bmd)

    params = [ps[key] for key in get_params(fit_method)]
    loglik = -tcpl_obj(ps=params, conc=conc, resp=resp, fit_method=get_fit_method(fit_method))

    # for bmd bounds, we want the difference between the max log-likelihood and the
    # bounds log-likelihood to be equal to chi-squared at 1-2*onesp (typically .9)
    # with one degree of freedom divided by two.
    return mll - loglik - chi2.ppf(1 - 2 * onesp, 1) / 2
