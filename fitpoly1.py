import numpy as np
from fit_method_helper import init_fit_method, fit

def fitpoly1(conc, resp, bidirectional=True, to_fit=False):
    fitmethod = "poly1"
    pars, sds, out, mmed, er_est, args = init_fit_method(fitmethod, conc, resp, bidirectional)
      
    if to_fit:
        conc_max = np.max(conc)
    
        # Starting parameters for the Model
        a0 = mmed / conc_max  # use largest response with desired directionality
        if a0 == 0:
            a0 = 0.01  # if 0, use a smallish number
        guess = [a0, er_est]  # linear coeff (a); set to run through the max resp at the max conc

        out = fit(conc, resp, fitmethod, pars, sds, out, args, guess)
    
    return out, to_fit