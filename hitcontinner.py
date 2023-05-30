import numpy as np
import pandas as pd
from scipy.stats import t

from toplikelihood import toplikelihood
from acy import poly1, poly2


def hitcontinner(conc, resp, top, cutoff, er, ps, fit_method, caikwt, mll):
    if fit_method == "none":
        return 0
    if fit_method == "hill":
        fname = "hillfn"
    else:
        fname = fit_method

    p1 = 1 - caikwt
    p2 = 1
    data = pd.DataFrame({'conc': conc, 'resp': resp})
    med_resp = data.groupby('conc')['resp'].median().reset_index()
    for y in med_resp["resp"]:
        p2 *= t.cdf((y - np.sign(top) * cutoff) / np.exp(er), 4) if top < 0 else 1 - t.cdf((y - np.sign(top) * cutoff) / np.exp(er), 4)

    p2 = 1 - p2

    ps = list(ps.values())
    ps = np.array([p for p in ps if not np.isnan(p)])
    p3 = toplikelihood(globals()[fname], cutoff, conc, resp, ps, top, mll)

    return p1 * p2 * p3
