import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from acy import tcplObj, cnst
from mad import mad

def fitcnst(conc, resp, nofit=False):
    pars = "er"
    myparams = ["success", "aic", "rme", "er"]
    
    if nofit:
        out = {param: np.nan for param in myparams}
        return out
    
    er_est = np.log(rmad) if (rmad := mad(resp)) > 0 else np.log(1e-32)

    try:
        #default method is bounded Brent
        offset = 2
        fname = "cnst"
        func = globals()[fname]
        fit = minimize_scalar(tcplObj, bracket=(er_est - offset, er_est + offset), method='brent', options={'maxiter': 500, 'xtol': 1e-4}, args=(conc, resp, func))
        if fit.success:
            success = 1
            er = fit.x
            aic = 2 - 2 * fit.fun # AIC=2k-2ln(L), with k = #estimated parameters, L = maximized value of likelihood function, where ln(L) = fit.fun?
            rme = np.sqrt(np.mean((np.zeros_like(resp) - resp) ** 2))
        else:
            success = 0
            er = np.nan
            aic = np.nan
            rme = np.nan
            
    except ValueError as error:
        print("An exception occurred:", error)
        success = 0
        er = np.nan
        aic = np.nan
        rme = np.nan
        
    return {"success": success, "aic": aic, "rme": rme, "er": er, "pars": pars}

