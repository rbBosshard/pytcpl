import numpy as np
from scipy.optimize import minimize_scalar

from acy import tcplObj
from acy import acy
from bmdobj import bmdobj

def bmdbounds(fit_method, bmr, pars, conc, resp, onesidedp=0.05, bmd=None, which_bound="lower"):

    if bmd is None:
        bmd = acy(bmr, pars, type=fit_method)
    if not np.isfinite(bmd):
        return np.nan

    if fit_method == "hill":
        fname = fit_method + "fn"
    else:
        fname = fit_method

    maxloglik = tcplObj(p=pars, conc=conc, resp=resp, fname=fname)

    if which_bound == "lower":
        xs = 10 ** np.linspace(-5, np.log10(bmd), num=100)
        ys = np.array([bmdobj(x, fname=fname, bmr=bmr, conc=conc, resp=resp, ps=pars, mll=maxloglik,
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
        ys = np.array([bmdobj(x, fname=fname, bmr=bmr, conc=conc, resp=resp, ps=pars, mll=maxloglik,
                              onesp=onesidedp, partype=2) for x in xs])
        if not np.any(ys >= 0) or not np.any(ys < 0):
            return np.nan
        bmdrange = np.array([bmd, np.min(xs[ys >= 0])])

    try:
        out = minimize_scalar(bmdobj, bracket=bmdrange, args=(fname, bmr, conc, resp, pars, maxloglik,
                                                              onesidedp, 2), method='brentq')
        if out.success:
            return out.root
        else:
            return np.nan
    except:
        return np.nan
