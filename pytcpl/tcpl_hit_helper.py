import numpy as np
import pandas as pd
from scipy.stats import t, chi2
from pytcpl.tcpl_obj_fn import tcpl_obj
from pytcpl.fit_models import get_fit_model


def hit_cont_inner(conc, resp, top, cutoff, er, ps, fit_model, caikwt, mll):
    # if fit_model in ["hill", "gnls"]:
    #     fit_model += "_"
    p1 = 1 - caikwt
    p2 = 1
    data = pd.DataFrame({'conc': conc, 'resp': resp})
    med_resp = data.groupby('conc')['resp'].median().reset_index()
    for y in med_resp["resp"]:
        p2 *= t.cdf((y - np.sign(top) * cutoff) / np.exp(er), 4) if top < 0 \
            else 1 - t.cdf((y - np.sign(top) * cutoff) / np.exp(er), 4)

    p2 = 1 - p2
    ps = list(ps.values())
    ps = np.array([p for p in ps if not np.isnan(p)])
    p3 = top_likelihood(fit_model, cutoff, conc, resp, ps, top, mll)

    return p1 * p2 * p3


def top_likelihood(fit_model, cutoff, conc, resp, ps, top, mll):
    # reparameterize so that top is exactly at cutoff
    if fit_model == "exp2":
        ps[0] = cutoff / (np.exp(np.max(conc) / ps[1]) - 1)
    elif fit_model == "exp3":
        ps[0] = cutoff / (np.exp((np.max(conc) / ps[1]) ** ps[2]) - 1)
    elif fit_model == "exp4":
        ps[0] = cutoff
    elif fit_model == "exp5":
        ps[0] = cutoff
    elif fit_model == "hill":
        ps[0] = cutoff
    elif fit_model == "gnls":
        ps[0] = cutoff
    elif fit_model == "poly1":
        ps[0] = cutoff / np.max(conc)
    elif fit_model == "poly2":
        ps[0] = cutoff / (np.max(conc) / ps[1] + (np.max(conc) / ps[1]) ** 2)
    elif fit_model == "pow":
        ps[0] = cutoff / (np.max(conc) ** ps[1])

    # get loglikelihood of top exactly at cutoff, use likelihood profile test
    loglik = tcpl_obj(ps=ps, conc=conc, resp=resp, fit_model=get_fit_model(fit_model))
    if abs(top) >= cutoff:
        out = (1 + chi2.cdf(2 * (mll - loglik), 1)) / 2
    else:
        out = (1 - chi2.cdf(2 * (mll - loglik), 1)) / 2

    return out


def nest_select(aics, mod1, mod2, dfdiff, pval=0.05):
    if np.isnan(aics[mod1]):
        loser = mod1  # if model 1 AIC is NaN, it is the loser
    elif aics[mod2] <= aics[mod1]:
        # if both AICs exist, model 2 AIC is lower, and model 2 passes the ratio test,
        # model 2 wins and model 1 loses; otherwise, model 2 loses.
        chisq = aics[mod1] - aics[mod2] + 2 * dfdiff  # 2 * loglikelihood(poly2) - 2 * loglikelihood(poly1)
        ptest = 1 - chi2.cdf(chisq, dfdiff)
        if ptest < pval:
            loser = mod1
        else:
            loser = mod2
    else:
        loser = mod2

    return {name: value for name, value in aics.items() if name != loser}
