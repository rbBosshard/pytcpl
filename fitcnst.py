from fit_method_helper import init_fit_method, fit

def fitcnst(conc, resp, bidirectional=True, to_fit=False):
    fitmethod = "cnst"
    pars, sds, out, mmed, er_est, args = init_fit_method(fitmethod, conc, resp)
    
    if to_fit:
        guess = [er_est]  # linear coeff (a); set to run through the max resp at the max conc
        out = fit(conc, resp, fitmethod, pars, sds, out, args, guess)
    
    return out, to_fit

        

