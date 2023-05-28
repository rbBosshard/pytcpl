import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize

from acy import tcplObj, cnst
from fit_method_helper import init_fit_method, generate_output

def fitcnst(conc, resp, nofit=False):
    fitmethod = "cnst"
    pars, sds, mmed, er_est, out = init_fit_method(fitmethod, conc, resp)
    if nofit:
        return out
    
    conc_max = np.max(conc)
    # Optimize the model
    offset = 2
    # Starting parameters for the Model
    a0 = mmed / conc_max  # use largest response with desired directionality
    if a0 == 0:
        a0 = 0.01  # if 0, use a smallish number
    guess = [er_est]  # linear coeff (a); set to run through the max resp at the max conc

    args = (conc, resp, globals()[fitmethod])
    try:
        # fit = minimize_scalar(tcplObj, bracket=(er_est - offset, er_est + offset), method='brent', options={'maxiter': 500, 'xtol': 1e-4}, args=args)
        fit = minimize(tcplObj, x0=guess, method = 'L-BFGS-B', args=args)
        print(f"{fitmethod} >>> Fitted.")
    except Exception as e:
        print(f"{fitmethod} >>> Error during optimization: {e}", )
        fit = None

    if fit:
        out = generate_output(fitmethod, conc, resp, pars, sds, out, fit)
        
    return out