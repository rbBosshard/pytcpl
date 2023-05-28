import numpy as np
from scipy.stats import t

from toplikelihood import toplikelihood


def hitcontinner(conc, resp, top, cutoff, er, ps, fit_method, caikwt, mll):
    if fit_method == "none":
        return 0
    if fit_method == "hill":
        fname = "hillfn"
    else:
        fname = fit_method

    P1 = 1 - caikwt
    P2 = 1
    med_resp = np.median(resp.groupby(conc)["resp"].median())
    for y in med_resp:
        P2 *= t.cdf((y - np.sign(top) * cutoff) / np.exp(er), 4) if top < 0 else 1 - t.cdf((y - np.sign(top) * cutoff) / np.exp(er), 4)

    P2 = 1 - P2

    ps = np.array([p for p in ps if not np.isnan(p)])
    P3 = toplikelihood(fname, cutoff, conc, resp, ps, top, mll)

    return P1 * P2 * P3
