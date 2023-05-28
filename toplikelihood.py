import numpy as np
from scipy.stats import chi2

from acy import tcplObj

def toplikelihood(fname, cutoff, conc, resp, ps, top, mll):
    if fname == "exp2":
        ps[0] = cutoff / (np.exp(np.max(conc) / ps[1]) - 1)
    elif fname == "exp3":
        ps[0] = cutoff / (np.exp((np.max(conc) / ps[1]) ** ps[2]) - 1)
    elif fname == "exp4":
        ps[0] = cutoff
    elif fname == "exp5":
        ps[0] = cutoff
    elif fname == "hillfn":
        ps[0] = cutoff
    elif fname == "gnls":
        ps[0] = cutoff
    elif fname == "poly1":
        ps[0] = cutoff / np.max(conc)
    elif fname == "poly2":
        ps[0] = cutoff / (np.max(conc) / ps[1] + (np.max(conc) / ps[1]) ** 2)
    elif fname == "pow":
        ps[0] = cutoff / (np.max(conc) ** ps[1])

    loglik = tcplObj(p=ps, conc=conc, resp=resp, fname=fname)
    if abs(top) >= cutoff:
        out = (1 + chi2.cdf(2 * (mll - loglik), 1)) / 2
    if abs(top) < cutoff:
        out = (1 - chi2.cdf(2 * (mll - loglik), 1)) / 2

    return out
