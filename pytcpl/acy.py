import numpy as np
from scipy.optimize import root_scalar
from fit_models import get_fit_model

def acy(y, modpars, fit_model, returntop=False, returntoploc=False, getloss=False, verbose=False):
    # unpack parameter dictionary into local variables
    if "pars" in modpars:
        locals().update(modpars["pars"])
    else:
        locals().update(modpars)

    if not returntop:
        if 'tp' in locals() and locals()["tp"] and abs(y) >= abs(locals()["tp"]):
            if verbose:
                print("y (specified activity response) is greater than tp in function acy, returning NA")
            return np.nan
        if 'top' in locals() and locals()["top"] and abs(y) >= abs(locals()["top"]):
            if verbose:
                print("y (specified activity response) is greater than top in function acy, returning NA")
            return np.nan
        if 'tp' in locals() and locals()["tp"] and y * locals()["tp"] < 0:
            if verbose:
                print("y (specified activity response) is wrong sign in function acy, returning NA")
            return np.nan

    if fit_model == "poly1":
        return y / locals()["a"]
    elif fit_model == "poly2":
        return locals()["b"] * (-1 + np.sqrt(1 + 4 * y / locals()["a"])) / 2
    elif fit_model == "pow":
        return (y / locals()["a"]) ** (1 / locals()["p"])
    elif fit_model == "exp2":
        return locals()["b"] * np.log(y / locals()["a"] + 1)
    elif fit_model == "exp3":
        return locals()["b"] * (np.log(y / locals()["a"] + 1)) ** (1 / locals()["p"])
    elif fit_model == "exp4":
        return -locals()["ga"] * np.log2(1 - y / locals()["tp"])
    elif fit_model == "exp5":
        return locals()["ga"] * (-np.log2(1 - y / locals()["tp"])) ** (1 / locals()["p"])
    elif fit_model == "hill":
        return locals()["ga"] / ((locals()["tp"] / y) - 1) ** (1 / locals()["p"])
    elif fit_model == "gnls":
        def gnlsderivobj(x, tp, ga, p, la, q):
            a = ga ** p
            b = la ** (-q)
            return b * q * x ** (q + p) + a * b * (q - p) * x ** q - a * p

        args = (locals()["tp"], locals()["ga"], locals()["p"], locals()["la"], locals()["q"])

        try:
            toploc = root_scalar(gnlsderivobj, bracket=[locals()["ga"], locals()["la"]], args=args).root
        except ValueError:
            if verbose:
                print("toploc could not be found numerically")
            topval = locals()["tp"]
            toploc = np.nan
        else:
            topval = get_fit_model("gnls")(list(args), toploc)

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

        def acgnlsobj(x, y, tp, ga, p, la, q):
            # y is desired y value
            return get_fit_model("gnls")([tp, ga, p, la, q], x) - y

        args = (y, locals()["tp"], locals()["ga"], locals()["p"], locals()["la"], locals()["q"])
        bracket = [toploc, 1e5] if getloss else [1e-8, toploc]
        output = root_scalar(acgnlsobj, bracket=bracket, args=args).root
        return output or np.nan

    return np.nan



