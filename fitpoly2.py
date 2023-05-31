import numpy as np
from fit_method_helper import init_fit_method, fit

def fitpoly2(conc, resp, bidirectional=True, to_fit=False):
    fitmethod = "poly2"
    pars, sds, out, mmed, er_est, args = init_fit_method(fitmethod, conc, resp, bidirectional)
      
    if to_fit:
        conc_max = np.max(conc)
    
        # Starting parameters for the Model
        a0 = mmed / conc_max 
        if a0 == 0:
            a0 = 0.01
        guess = [a0 / 2, conc_max, er_est]

        out = fit(conc, resp, fitmethod, pars, sds, out, args, guess)
    
    return out, to_fit
