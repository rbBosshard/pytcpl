import numpy as np
import scipy.optimize as optimize
from scipy.stats import chi2

from acy import acy
from fit_models import get_params, get_fit_model
from tcpl_obj_fn import tcpl_obj


def bmd_bounds(fit_model, bmr, pars, conc, resp, onesidedp=0.05, bmd=None, which_bound="lower"):
    """
    BMD Bounds

    """

    # calculate bmd, if necessary
    if bmd is None:
        bmd = acy(bmr, pars, fit_model=fit_model)
    if bmd is None or not np.isfinite(bmd):
        return None

    params = [pars[key] for key in get_params(fit_model)]

    # negated minimized negative loglikelihood. Todo: recheck if everything is correct like this
    maxloglik = -tcpl_obj(params=params, conc=conc, resp=resp, fit_model=get_fit_model(fit_model))

    # search for bounds to ensure sign change
    bmdrange = None
    if which_bound == "lower":
        xs = 10 ** np.linspace(-5, np.log10(bmd), num=100)
        ys = np.array([bmd_obj(x, fit_model=fit_model, bmr=bmr, conc=conc, resp=resp, ps=pars, mll=maxloglik,
                               onesp=onesidedp, partype=2) for x in xs])
        if not np.any(ys >= 0) or not np.any(ys < 0):
            return None
        bmdrange = np.array([np.max(xs[ys >= 0]), bmd])

    elif which_bound == "upper":
        if fit_model == "gnls":
            toploc = acy(bmr, pars, fit_model="gnls", returntoploc=True)
            xs = 10 ** np.linspace(np.log10(bmd), np.log10(toploc), num=100)
        else:
            xs = 10 ** np.linspace(np.log10(bmd), 5, num=100)
        ys = np.array([bmd_obj(x, fit_model=fit_model, bmr=bmr, conc=conc, resp=resp, ps=pars, mll=maxloglik,
                               onesp=onesidedp, partype=2) for x in xs])
        if not np.any(ys >= 0) or not np.any(ys < 0):
            return None
        bmdrange = np.array([bmd, np.min(xs[ys >= 0])])

    try:
        # use type 2 param. only
        return optimize.root_scalar(bmd_obj, bracket=bmdrange,
                                    args=(fit_model, bmr, conc, resp, pars, maxloglik, onesidedp, 2)).root
    except ValueError:
        return None


def bmd_obj(bmd, fit_model, bmr, conc, resp, ps, mll, onesp, partype=2):
    # implements the BMD substitutions in Appendix A of the Technical Report.
    # Changes one of the existing parameters to an explicit bmd parameter through
    # the magic of algebra.
    if fit_model == "exp2":
        if partype == 1:
            ps["a"] = bmr / (np.exp(bmd / ps["b"]) - 1)
        elif partype == 2:
            ps["b"] = bmd / (np.log(bmr / ps["a"] + 1))
        elif partype == 3:
            ps["b"] = bmd / (np.log(bmr / ps["a"] + 1))
    elif fit_model == "exp3":
        if partype == 1:
            ps["a"] = bmr / (np.exp((bmd / ps["b"]) ** ps["p"]) - 1)
        elif partype == 2:
            ps["b"] = bmd / (np.log(bmr / ps["a"] + 1)) ** (1 / ps["p"])
        elif partype == 3:
            ps["p"] = np.log(np.log(bmr / ps["a"] + 1)) / np.log(bmd / ps["b"])
    elif fit_model == "exp4":
        if partype == 1:
            ps["tp"] = bmr / (1 - 2 ** (-bmd / ps["ga"]))
        elif partype == 2:
            ps["ga"] = bmd / (-np.log2(1 - bmr / ps["tp"]))
        elif partype == 3:
            ps["ga"] = bmd / (-np.log2(1 - bmr / ps["tp"]))
    elif fit_model == "exp5":
        if partype == 1:
            ps["tp"] = bmr / (1 - 2 ** (-(bmd / ps["ga"]) ** ps["p"]))
        elif partype == 2:
            ps["ga"] = bmd / ((-np.log2(1 - bmr / ps["tp"])) ** (1 / ps["p"]))
        elif partype == 3:
            ps["p"] = np.log(-np.log2(1 - bmr / ps["tp"])) / np.log(bmd / ps["ga"])
    elif fit_model == "hill":
        if partype == 1:
            ps["tp"] = bmr * (1 + (ps["ga"] / bmd) ** ps["p"])
        elif partype == 2:
            ps["ga"] = bmd * (ps["tp"] / bmr - 1) ** (1 / ps["p"])
        elif partype == 3:
            ps["p"] = np.log(ps["tp"] / bmr - 1) / np.log(ps["ga"] / bmd)
    elif fit_model == "gnls":
        if partype == 1:
            ps["tp"] = bmr * ((1 + (ps["ga"] / bmd) ** ps["p"]) * (1 + (bmd / ps["la"]) ** ps["q"]))
        elif partype == 2:
            ps["ga"] = bmd * ((ps["tp"] / (bmr * (1 + (bmd / ps["la"]) ** ps["q"]))) - 1) ** (1 / ps["p"])
        elif partype == 3:
            ps["p"] = np.log(ps["tp"] / (bmr * (1 + (bmd / ps["la"]) ** ps["q"]))) - 1 / np.log(ps["ga"] / bmd)
    elif fit_model == "poly1":
        if partype == 1:
            ps["a"] = bmr / bmd
        elif partype == 2:
            ps["a"] = bmr / bmd
        elif partype == 3:
            ps["a"] = bmr / bmd
    elif fit_model == "poly2":
        if partype == 1:
            ps["a"] = bmr / (bmd / ps["b"] + (bmd / ps["b"]) ** 2)
        elif partype == 2:
            ps["b"] = 2 * bmd / (np.sqrt(1 + 4 * bmr / ps["a"]) - 1)
        elif partype == 3:
            ps["b"] = 2 * bmd / (np.sqrt(1 + 4 * bmr / ps["a"]) - 1)
    elif fit_model == "pow":
        if partype == 1:
            ps["a"] = bmr / (bmd ** ps["p"])
        elif partype == 2:
            ps["a"] = bmr / (bmd ** ps["p"])
        elif partype == 3:
            ps["p"] = np.log(bmr / ps["a"]) / np.log(bmd)

    params = [ps[key] for key in get_params(fit_model)]
    loglik = -tcpl_obj(params=params, conc=conc, resp=resp, fit_model=get_fit_model(fit_model))

    # for bmd bounds, we want the difference between the max log-likelihood and the
    # bounds log-likelihood to be equal to chi-squared at 1-2*onesp (typically .9)
    # with one degree of freedom divided by two.
    return mll - loglik - chi2.ppf(1 - 2 * onesp, 1) / 2
