import numpy as np
from scipy.optimize import minimize
import yaml

from acy import poly1
from acy import tcplObj

from fit_method_helper import init_fit_method, generate_output

def fitpoly1(conc, resp, bidirectional=True, verbose=False, nofit=False):
    fitmethod = "poly1"
    pars, sds, mmed, er_est, out = init_fit_method(fitmethod, conc, resp, bidirectional)
    if nofit:
        return out
      
    conc_max = np.max(conc)
    
    # Starting parameters for the Model
    a0 = mmed / conc_max  # use largest response with desired directionality
    if a0 == 0:
        a0 = 0.01  # if 0, use a smallish number
    guess = [a0, er_est]  # linear coeff (a); set to run through the max resp at the max conc

    # # Generate the bound matrices to constrain the model.
    # Ui = np.array([[1, 0], [-1, 0]], dtype=np.float64)

    # if not bidirectional:
    #     bnds = [0, -1e8 * np.abs(a0)]  # a bounds (always positive)
    # else:
    #     bnds = [-1e8 * np.abs(a0), -1e8 * np.abs(a0)]  # a bounds (positive or negative) # is this correct?

    # Ci = np.array(bnds, dtype=np.float64).reshape((2, 1))

    # Optimize the model
    args = (conc, resp, globals()[fitmethod])
    try:
        fit = minimize(tcplObj, x0=guess, method = 'L-BFGS-B', args=args)
    except Exception as e:
        print(f"{fitmethod} >>> Error during optimization: {e} {fit.message}")
        fit = None

    # Generate some summary statistics
    if fit:
        out = generate_output(fitmethod, conc, resp, pars, sds, out, fit)

    return out