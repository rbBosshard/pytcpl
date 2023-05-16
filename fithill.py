import numpy as np
from scipy.optimize import minimize
from scipy.linalg import inv

from acy import tcplObj

def fithill(conc, resp, bidirectional=True, verbose=False, nofit=False):
    logc = np.log10(conc)
    fenv = globals()
  
    pars = ["tp", "ga", "p", "er"]
    sds = [param + "_sd" for param in pars]
    myparams = ["success", "aic", "cov", "rme", "modl"] + pars + sds + ["pars", "sds"]
  
    if nofit:
        out = {param: np.nan for param in myparams}
        out["success"] = out["cov"] = np.nan
        out["pars"] = pars
        out["sds"] = sds
        return out
    
    rmds = np.array([np.median(resp[logc == logc_val]) for logc_val in np.unique(logc)])
    if not bidirectional:
        mmed = rmds[np.argmax(rmds)]
    else:
        mmed = rmds[np.argmax(np.abs(rmds))]
    mmed_conc = float(np.unique(logc)[np.argmax(rmds)])
    
    resp_max = np.max(resp)
    resp_min = np.min(resp)
    logc_min = np.min(logc)
    logc_max = np.max(logc)
    
    er_est = np.log(np.abs(np.median(resp - np.median(resp))))

    g = [mmed, mmed_conc - 0.5, 1.2, er_est]
    if g[0] == 0:
        g[0] = 0.1
    
    Ui = np.array([[1, 0, 0, 0],
                   [-1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, -1, 0]])
  
    if not bidirectional:
        hbnds = [0, -1.2 * resp_max, logc_min - 1, -(logc_max + 0.5), 0.3, -8]
    else:
        val = 1.2 * max(abs(resp_min), abs(resp_max))
        # hbnds = [-val, -val, logc_min - 1, -(logc_max + 
        hbnds = np.array(hbnds).reshape(-1, 1)

  
    fit = minimize(tcplObj, g, args=(logc, resp), method="Nelder-Mead", bounds=hbnds)
  
    if fit.success:
        if verbose:
            print("hill >>>", fit.nfev, fit.status)
  
        success = 1
        aic = 2 * len(fit.x) - 2 * fit.fun
        fname = "loghill"
        func = globals()[fname]
        modl = func(fit.x, np.log10(conc))
        rme = np.sqrt(np.mean((modl - resp) ** 2))
  
        tp, ga, p, er = fit.x
  
        try:
            cov = inv(fit.hess_inv)
            hdiag_sqrt = np.sqrt(np.diag(cov))
            sds = hdiag_sqrt if not np.any(np.isnan(hdiag_sqrt)) else [np.nan] * len(pars)
            ga_sd *= ga * np.log(10)
        except np.linalg.LinAlgError:
            cov = np.nan
            sds = [np.nan] * len(pars)
  
    else:
        success = 0
        aic = np.nan
        cov = np.nan
        rme = np.nan
        modl = np.nan
        tp = ga = p = er = np.nan
        sds = [np.nan] * len(pars)
  
    out = {
        "success": success,
        "aic": aic,
        "cov": cov,
        "rme": rme,
        "modl": modl,
    }
  
    for i, param in enumerate(pars):
        out[param] = eval(param)
        out[param + "_sd"] = sds[i]
  
    out["pars"] = pars
    out["sds"] = sds
  
    return out