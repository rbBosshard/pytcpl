import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

from pytcpl.acy import tcplObj
from pytcpl.acy import acy


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

    if bmd is None:
        bmd = acy(bmr, pars, type=fit_method)
    if not np.isfinite(bmd):
        return np.nan

    if fit_method == "hill":
        fname = fit_method + "fn"
    else:
        fname = fit_method

    maxloglik = tcplObj(ps=pars, conc=conc, resp=resp, fname=fname)

    if which_bound == "lower":
        xs = 10 ** np.linspace(-5, np.log10(bmd), num=100)
        ys = np.array([bmd_obj(x, fname=fname, bmr=bmr, conc=conc, resp=resp, ps=pars, mll=maxloglik,
                               onesp=onesidedp, partype=2) for x in xs])
        if not np.any(ys >= 0) or not np.any(ys < 0):
            return np.nan
        bmdrange = np.array([np.max(xs[ys >= 0]), bmd])
    elif which_bound == "upper":
        if fit_method == "gnls":
            toploc = acy(bmr, pars, type="gnls", returntoploc=True)
            xs = 10 ** np.linspace(np.log10(bmd), np.log10(toploc), num=100)
        else:
            xs = 10 ** np.linspace(np.log10(bmd), 5, num=100)
        ys = np.array([bmd_obj(x, fname=fname, bmr=bmr, conc=conc, resp=resp, ps=pars, mll=maxloglik,
                               onesp=onesidedp, partype=2) for x in xs])
        if not np.any(ys >= 0) or not np.any(ys < 0):
            return np.nan
        bmdrange = np.array([bmd, np.min(xs[ys >= 0])])

    try:
        # Change?
        out = minimize_scalar(bmd_obj, bracket=bmdrange, args=(fname, bmr, conc, resp, pars, maxloglik,
                                                               onesidedp, 2), method='brentq')
        if out.success:
            return out.root
        else:
            return np.nan
    except:
        return np.nan


def bmd_obj(bmd, fname, bmr, conc, resp, ps, mll, onesp, partype=2):
    def log2(x):
        return np.log(x) / np.log(2)

    if fname == "exp2":
        if partype == 1:
            ps["a"] = bmr / (np.exp(bmd / ps["b"]) - 1)
        elif partype == 2:
            ps["b"] = bmd / (np.log(bmr / ps["a"] + 1))
        elif partype == 3:
            ps["b"] = bmd / (np.log(bmr / ps["a"] + 1))
    elif fname == "exp3":
        if partype == 1:
            ps["a"] = bmr / (np.exp((bmd / ps["b"]) ** ps["p"]) - 1)
        elif partype == 2:
            ps["b"] = bmd / (np.log(bmr / ps["a"] + 1)) ** (1 / ps["p"])
        elif partype == 3:
            ps["p"] = np.log(np.log(bmr / ps["a"] + 1)) / np.log(bmd / ps["b"])
    elif fname == "exp4":
        if partype == 1:
            ps["tp"] = bmr / (1 - 2 ** (-bmd / ps["ga"]))
        elif partype == 2:
            ps["ga"] = bmd / (-log2(1 - bmr / ps["tp"]))
        elif partype == 3:
            ps["ga"] = bmd / (-log2(1 - bmr / ps["tp"]))
    elif fname == "exp5":
        if partype == 1:
            ps["tp"] = bmr / (1 - 2 ** (-(bmd / ps["ga"]) ** ps["p"]))
        elif partype == 2:
            ps["ga"] = bmd / ((-log2(1 - bmr / ps["tp"])) ** (1 / ps["p"]))
        elif partype == 3:
            ps["p"] = np.log(-log2(1 - bmr / ps["tp"])) / np.log(bmd / ps["ga"])
    elif fname == "hillfn":
        if partype == 1:
            ps["tp"] = bmr * (1 + (ps["ga"] / bmd) ** ps["p"])
        elif partype == 2:
            ps["ga"] = bmd * (ps["tp"] / bmr - 1) ** (1 / ps["p"])
        elif partype == 3:
            ps["p"] = np.log(ps["tp"] / bmr - 1) / np.log(ps["ga"] / bmd)
    if fname == "gnls":
        if partype == 1:
            ps["tp"] = bmr * ((1 + (ps["ga"] / bmd) ** ps["p"]) * (1 + (bmd / ps["la"]) ** ps["q"]))
        elif partype == 2:
            ps["ga"] = bmd * ((ps["tp"] / (bmr * (1 + (bmd / ps["la"]) ** ps["q"]))) - 1) ** (1 / ps["p"])
        elif partype == 3:
            ps["p"] = np.log(ps["tp"] / (bmr * (1 + (bmd / ps["la"]) ** ps["q"]))) - 1 / np.log(ps["ga"] / bmd)
    elif fname == "poly1":
        if partype == 1:
            ps["a"] = bmr / bmd
        elif partype == 2:
            ps["a"] = bmr / bmd
        elif partype == 3:
            ps["a"] = bmr / bmd
    elif fname == "poly2":
        if partype == 1:
            ps["a"] = bmr / (bmd / ps["b"] + (bmd / ps["b"]) ** 2)
        elif partype == 2:
            ps["b"] = 2 * bmd / (np.sqrt(1 + 4 * bmr / ps["a"]) - 1)
        elif partype == 3:
            ps["b"] = 2 * bmd / (np.sqrt(1 + 4 * bmr / ps["a"]) - 1)
    elif fname == "pow":
        if partype == 1:
            ps["a"] = bmr / (bmd ** ps["p"])
        elif partype == 2:
            ps["a"] = bmr / (bmd ** ps["p"])
        elif partype == 3:
            ps["p"] = np.log(bmr / ps["a"]) / np.log(bmd)

    loglik = tcplObj(ps=ps, conc=conc, resp=resp, fname=fname)
    return mll - loglik - chi2.ppf(1 - 2 * onesp, 1) / 2
