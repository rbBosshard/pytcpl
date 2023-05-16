import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from acy import tcplObj, cnst
from mad import mad

def fitcnst(conc, resp, nofit=False):
    myparams = ["success", "aic", "rme", "er"]
  
    if nofit:
        out = {param: np.nan for param in myparams}
        return out
    
    er_est = np.log(rmad) if (rmad := mad(resp)) > 0 else np.log(1e-32)
    # def tcplObj(p, conc=conc, resp=resp):
    #     mu = cnst(p, conc)
    #     if np.any(np.isnan(mu)):
    #         return np.inf
    #     else:
    #         err = np.exp(p)
    #         return -np.sum(np.log(norm.pdf((resp - mu) / err)) - np.log(err))
    
    try:
        #default method is bounded Brent
        offset = 2
        fname = "cnst"
        func = globals()[fname]
        # ps = {"a": 1, "tp":2, "b":4, "ga":3, "p":4, "la":4, "q":4, "er":0.5}
        # p = list(ps.values()) 
        # fout = tcplObj(p, conc, resp, func)
        fit = minimize_scalar(tcplObj, bracket=(er_est - offset, er_est + offset), method='Brent', options={'maxiter': 500}, args=(conc, resp, func))
        # print(fit)
        if fit.success:
            success = 1
            er = fit.x
            aic = 2 - 2 * fit.fun
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
        
    return {"success": success, "aic": aic, "rme": rme, "er": er, "pars": "er"}

