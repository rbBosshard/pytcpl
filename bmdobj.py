import numpy as np
from scipy.stats import chi2

from acy import tcplObj

def bmdobj(bmd, fname, bmr, conc, resp, ps, mll, onesp, partype=2):
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
