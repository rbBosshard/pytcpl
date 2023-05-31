import numpy as np
import yaml
from mad import mad
from acy import cnst, poly1, poly2

from scipy.optimize import minimize

from acy import tcplObj


def fit(conc, resp, fitmethod, pars, sds, out, args, guess):
    try:
        # fit = minimize_scalar(tcplObj, bracket=(er_est - offset, er_est + offset), method='brent', options={'maxiter': 500, 'xtol': 1e-4}, args=args)
        fit = minimize(tcplObj, x0=guess, method='L-BFGS-B', args=args)
    except Exception as e:
        print(f"{fitmethod} >>> Error during optimization: {e}", )
        fit = None

    if fit:
        out = generate_output(fitmethod, conc, resp, pars, sds, out, fit)
    return out

def init_fit_method(fitmethod, conc, resp, bidirectional=False):
    # with open("config.yaml", "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    #     pars_keys = config['FITPARS'][fitmethod.upper()]

    config = {
    'cnst': ['er'],
    'exp2': ['a', 'b', 'er'],
    'exp3': ['a', 'b', 'p', 'er'],
    'exp4': ['tp', 'ga', 'er'],
    'exp5': ['tp', 'ga', 'p', 'er'],
    'hill': ['tp', 'ga', 'p', 'er'],
    'poly1': ['a', 'er'],
    'poly2': ['a', 'b', 'er'],
    'pow': ['a', 'p', 'er'],
    'gnls': ['tp', 'ga', 'p', 'la', 'q', 'er']
    }

    pars_keys = config[fitmethod]

    pars = dict.fromkeys(pars_keys)
    sds = dict.fromkeys([param + "_sd" for param in pars])
    myparams = ["success", "aic", "cov", "rme", "modl", "pars", "sds"]
    out = {param: np.nan for param in myparams}
    out["pars"] = pars
    out["sds"] = sds

    # median at each conc, for multi-valued responses
    unique_conc = np.unique(conc)
    rmds = np.array([np.median(resp[conc == c]) for c in unique_conc])

    # get max response and corresponding conc
    mmed = rmds[np.argmax(rmds)] if not bidirectional else rmds[np.argmax(np.abs(rmds))]

    er_est = np.log(rmad) if (rmad := mad(resp)) > 0 else np.log(1e-16)
    args = (conc, resp, globals()[fitmethod])
    
    return pars, sds, out, mmed, er_est, args


def generate_output(fitmethod, conc, resp, pars, sds, out, fit):
    out["success"] = 1
    out["aic"] = 2 * len(fit.x) + 2 * fit.fun
    out["pars"] = {param: fit.x[i] for i, param in enumerate(pars)}
    out["modl"] = globals()[fitmethod](fit.x, conc)
    out["rme"] = np.sqrt(np.mean((out["modl"] - resp) ** 2))
    
    try:
        # Estimate the covariance matrix using the inverse of the Hessian
        # Inverse of the objective function’s Hessian; may be an approximation. Not available for all solvers.
        covariance_matrix = np.linalg.inv(fit.hess_inv.todense())
        # Access the uncertainty estimates
        uncertainties = np.sqrt(np.diag(covariance_matrix)) # uncertainties = standard deviations of parameters = diag_sqrt of covariance matrix

        if not np.any(np.isnan(uncertainties)):
            out["cov"] = 1
            out["sds"] = {param: uncertainties[i] for i, param in enumerate(sds)}

    except Exception as e: 
            print(f"{fitmethod} >>> Error calculating parameter covariance: {e}")
            out["cov"] = 0

    return out