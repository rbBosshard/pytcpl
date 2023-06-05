import warnings
import math
from scipy.optimize import root_scalar




def acy(y, modpars, type="hill", returntop=False, returntoploc=False, getloss=False, verbose=False):
    if "pars" in modpars:
        locals().update(modpars["pars"])
    else:
        locals().update(modpars)  # unpack modpars dict into local variables

    if not returntop:
        if "tp" in modpars and modpars["tp"] is not None and abs(y) >= abs(modpars["tp"]):
            if verbose:
                warnings.warn("y (specified activity response) is greater than tp in function acy, returning NA")
            return math.nan
        if "top" in modpars and modpars["top"] is not None and abs(y) >= abs(modpars["top"]):
            if verbose:
                warnings.warn("y (specified activity response) is greater than top in function acy, returning NA")
            return math.nan
        if "tp" in modpars and modpars["tp"] is not None and y * modpars["tp"] < 0:
            if verbose:
                warnings.warn("y (specified activity response) is wrong sign in function acy, returning NA")
            return math.nan

    if type == "poly1":
        return y / locals()["a"]
    elif type == "poly2":
        return locals()["b"] * (-1 + math.sqrt(1 + 4 * y / locals()["a"])) / 2
    elif type == "pow":
        return (y / locals()["a"]) ** (1 / locals()["p"])
    elif type == "exp2":
        return locals()["b"] * math.log(y / locals()["a"] + 1)
    elif type == "exp3":
        return locals()["b"] * (math.log(y / locals()["a"] + 1)) ** (1 / locals()["p"])
    elif type == "exp4":
        return -locals()["ga"] * math.log2(1 - y / locals()["tp"])
    elif type == "exp5":
        return locals()["ga"] * (-math.log2(1 - y / locals()["tp"])) ** (1 / locals()["p"])
    elif type == "hill":
        return locals()["ga"] / ((locals()["tp"] / y) - 1) ** (1 / locals()["p"])
    elif type == "gnls":
        toploc = None
        args = (locals()["tp"], locals()["ga"], locals()["p"], locals()["la"], locals()["q"])
        try:
            toploc = root_scalar(gnlsderivobj, args=args, method='brentq',
                                 bracket=[locals()["ga"], locals()["la"]]).root
        except ValueError:
            if verbose:
                warnings.warn("toploc could not be found numerically")
            topval = locals()["tp"]
            toploc = math.nan
        else:
            topval = gnls(list(args), toploc)

        if returntoploc:
            return toploc
        if returntop:
            return topval

        if abs(y) > abs(topval):
            if verbose:
                warnings.warn("y is greater than gnls top in function acy, returning NA")
            return math.nan
        if y == topval:
            return toploc

        if getloss:
            args = (y, locals()["tp"], locals()["ga"], locals()["p"], locals()["la"], locals()["q"])
            output = root_scalar(acgnlsobj, args=args, method='brentq', bracket=[toploc, 1e5]).root
        else:
            output = root_scalar(acgnlsobj, args=args, method='brentq', bracket=[1e-8, toploc]).root

        if math.isnan(output):
            return math.nan
        else:
            return output

    return math.nan


def gnlsderivobj(x, tp, ga, p, la, q):
    a = ga ** p
    b = la ** (-q)
    return b * q * x ** (q + p) + a * b * (q - p) * x ** q - a * p


def acgnlsobj(x, y, tp, ga, p, la, q):
    # y is desired y value
    return gnls([tp, ga, p, la, q], x) - y


#################
# Model Functions
import numpy as np
from scipy.stats import t, norm


def tcplObj(ps, conc, resp, fname, errfun="dt4"):
    mu = fname(ps=ps, x=conc)  # get model values for each conc, ps = parameter vector
    err = np.exp(ps[-1])  # last parameter is the log of the error variance
    # objective function is the sum of log-likelihood of response given the model at each concentration scaled by variance (err) -> normalize residulas (take into consideration the relative importance of each residual in the objective function)
    # Wrapped with scipy.optimize.minimize(): It is the convention that we call the optimization objective function a "cost function" or "loss function" and therefore, we want to minimize them, rather than maximize them, and hence the negative log likelihood is formed, rather than positive likelihood in your word
    # https://stats.stackexchange.com/questions/260505/why-do-we-use-negative-log-likelihood-to-estimate-parameters-for-ml
    if errfun == "dt4":
        nll = -np.sum(t.logpdf((resp - mu) / err, df=4) - np.log(
            err))  # degree of freedom paramter = 4, for Student’s t probability density function
    elif errfun == "dnorm":
        nll = -np.sum(norm.logpdf((resp - mu) / err) - np.log(err))
    return nll  # negative log likelihood scaled by variance


def cnst(ps, x):
    # ignores ps
    return np.zeros(len(x))


def exp2(ps, x):
    # a = ps[0], b = ps[1]
    return ps[0] * (np.exp(x / ps[1]) - 1)


def exp3(ps, x):
    # a = ps[0], b = ps[1], p = ps[2]
    return ps[0] * (np.exp((x / ps[1]) ** ps[2]) - 1)


def exp4(ps, x):
    # tp = ps[0], ga = ps[1]
    return ps[0] * (1 - 2 ** (-x / ps[1]))


def exp5(ps, x):
    # tp = ps[0], ga = ps[1], p = ps[2]
    return ps[0] * (1 - 2 ** (-(x / ps[1]) ** ps[2]))


def gnls(ps, x):
    # gnls function with regular units
    # tp = ps[0], ga = ps[1], p = ps[2], la = ps[3], q = ps[4]
    gn = 1 / (1 + (ps[1] / x) ** ps[2])
    ls = 1 / (1 + (x / ps[3]) ** ps[4])
    return ps[0] * gn * ls


def loggnls(ps, x):
    # gnls function with log units: x = log10(conc) and ga/la = log10(gain/loss ac50)
    # tp = ps[0], ga = ps[1], p = ps[2], la = ps[3], q = ps[4]
    gn = 1 / (1 + 10 ** ((ps[1] - x) * ps[2]))
    ls = 1 / (1 + 10 ** ((x - ps[3]) * ps[4]))
    return ps[0] * gn * ls


def hillfn(ps, x):
    # hill function with regular units
    # tp = ps[0], ga = ps[1], p = ps[2]
    return ps[0] / (1 + (ps[1] / x) ** ps[2])


def loghill(ps, x):
    # hill function with log units: x = log10(conc) and ga = log10(ac50)
    # tp = ps[0], ga = ps[1], p = ps[2]
    return ps[0] / (1 + 10 ** (ps[2] * (ps[1] - x)))


def poly1(ps, x):
    # a = ps[0]
    return ps[0] * x


def poly2(ps, x):
    # a = ps[0], b = ps[1]
    x0 = x / ps[1]
    return ps[0] * (x0 + x0 * x0)


def pow(ps, x):
    # a = ps[0], p = ps[1]
    return ps[0] * x ** ps[1]
