import numpy as np
from scipy.stats import chi2

def nestselect(aics, mod1, mod2, dfdiff, pval=0.05):
    if np.isnan(aics[mod1]):
        loser = mod1  # if model 1 AIC is NaN, it is the loser
    elif np.logical_and(aics[mod2] <= aics[mod1], aics[mod2] <= aics[mod1] + 2 * dfdiff):
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
