import numpy as np
from scipy.optimize import root_scalar
from scipy.stats import t, norm


def acy(y, modpars, fit_method, returntop=False, returntoploc=False, getloss=False, verbose=False):
    # unpack parameter dictionary into local variables
    if "pars" in modpars:
        locals().update(modpars["pars"])
    else:
        locals().update(modpars)

    if not returntop:
        if 'tp' in locals() and locals()["tp"] and abs(y) >= abs(locals()["tp"]):
            if verbose:
                print("y (specified activity response) is greater than tp in function acy, returning NA")
            return None
        if 'top' in locals() and locals()["top"] and abs(y) >= abs(locals()["top"]):
            if verbose:
                print("y (specified activity response) is greater than top in function acy, returning NA")
            return None
        if 'tp' in locals() and locals()["tp"] and y * locals()["tp"] < 0:
            if verbose:
                print("y (specified activity response) is wrong sign in function acy, returning NA")
            return None

    if fit_method == "poly1":
        return y / locals()["a"]
    elif fit_method == "poly2":
        return locals()["b"] * (-1 + np.sqrt(1 + 4 * y / locals()["a"])) / 2
    elif fit_method == "pow":
        return (y / locals()["a"]) ** (1 / locals()["p"])
    elif fit_method == "exp2":
        return locals()["b"] * np.log(y / locals()["a"] + 1)
    elif fit_method == "exp3":
        return locals()["b"] * (np.log(y / locals()["a"] + 1)) ** (1 / locals()["p"])
    elif fit_method == "exp4":
        return -locals()["ga"] * np.log2(1 - y / locals()["tp"])
    elif fit_method == "exp5":
        return locals()["ga"] * (-np.log2(1 - y / locals()["tp"])) ** (1 / locals()["p"])
    elif fit_method == "hill":
        return locals()["ga"] / ((locals()["tp"] / y) - 1) ** (1 / locals()["p"])
    elif fit_method == "gnls":
        args = (locals()["tp"], locals()["ga"], locals()["p"], locals()["la"], locals()["q"])
        try:
            toploc = root_scalar(gnlsderivobj, bracket=[locals()["ga"], locals()["la"]], args=args).root
        except ValueError:
            if verbose:
                print("toploc could not be found numerically")
            topval = locals()["tp"]
            toploc = np.nan
        else:
            topval = gnls_(list(args), toploc)

        if returntoploc:
            return toploc
        if returntop:
            return topval

        if abs(y) > abs(topval):
            if verbose:
                print("y is greater than gnls top in function acy, returning NA")
            return np.nan

        if y == topval:
            return toploc

        args = (y, locals()["tp"], locals()["ga"], locals()["p"], locals()["la"], locals()["q"])
        if getloss:
            output = root_scalar(acgnlsobj, bracket=[toploc, 1e5], args=args).root
        else:
            output = root_scalar(acgnlsobj, bracket=[1e-8, toploc], args=args).root

        return output or np.nan

    return np.nan


def gnlsderivobj(x, tp, ga, p, la, q):
    a = ga ** p
    b = la ** (-q)
    return b * q * x ** (q + p) + a * b * (q - p) * x ** q - a * p


def acgnlsobj(x, y, tp, ga, p, la, q):
    # y is desired y value
    return gnls_([tp, ga, p, la, q], x) - y


def tcpl_obj(ps, conc, resp, fit_method, errfun="dt4"):
    # Optimization objective function is called "cost function" or "loss function"
    # and therefore, we want to minimize them, rather than maximize them,
    # hence the negative log likelihood is formed, wrapped with scipy.optimize.minimize()
    # https://stats.stackexchange.com/questions/260505/why-do-we-use-negative-log-likelihood-to-estimate-parameters-for-ml

    # Objective function is the sum of log-likelihood of response
    # given the model at each concentration scaled by variance (err)
    mu = fit_method(ps=ps, x=conc)  # ps = parameter vector, get model values for each conc,
    err = np.exp(ps[-1])  # last parameter is the log of the error/variance
    residuals = (resp - mu) / err
    # Todo: try another density function
    if errfun == "dt4":
        # degree of freedom parameter = 4, for Student’s t probability density function
        nll = -np.sum(t.logpdf(residuals, df=4) - np.log(err))
    else:  # errfun == "dnorm":
        nll = -np.sum(norm.logpdf(residuals) - np.log(err))

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


def gnls_(ps, x):
    # gnls function with regular units
    # tp = ps[0], ga = ps[1], p = ps[2], la = ps[3], q = ps[4]
    gn = 1 / (1 + (ps[1] / x) ** ps[2])
    ls = 1 / (1 + (x / ps[3]) ** ps[4])
    return ps[0] * gn * ls


def gnls(ps, x):
    # gnls function with log units: x = log10(conc) and ga/la = log10(gain/loss ac50)
    # tp = ps[0], ga = ps[1], p = ps[2], la = ps[3], q = ps[4]
    if not isinstance(ps, dict):
        gn = 1 / (1 + 10 ** ((ps[1] - x) * ps[2]))
        ls = 1 / (1 + 10 ** ((x - ps[3]) * ps[4]))
        return ps[0] * gn * ls


def hill_(ps, x):
    # hill function with regular units
    # tp = ps[0], ga = ps[1], p = ps[2]
    return ps[0] / (1 + (ps[1] / x) ** ps[2])


def hill(ps, x):
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
