import numpy as np
from scipy.optimize import minimize

from acy import poly2
from acy import tcplObj
from mad import mad

def fitpoly2(conc, resp, bidirectional=True, verbose=False, nofit=False):
    fenv = globals()
    # Initialize myparams
    pars = [f"a{i}" for i in range(3)]
    sds = [f"{param}_sd" for param in pars]
    myparams = ["success", "aic", "cov", "rme", "modl"] + pars + sds + ["pars", "sds"]

    # Return myparams with appropriate NAs
    if nofit:
        out = {param: np.nan for param in myparams}
        out["success"] = out["cov"] = None
        out["pars"] = pars
        out["sds"] = sds
        return out

    conc = np.array(conc)
    resp = np.array(resp)

    # Median at each conc, for multi-valued responses
    unique_conc = np.unique(conc)
    rmds = np.array([np.median(resp[conc == c]) for c in unique_conc])

    # Get max response and corresponding conc
    if not bidirectional:
        mmed = rmds[np.argmax(rmds)]
    else:
        mmed = rmds[np.argmax(np.abs(rmds))]

    conc_max = np.max(conc)

    er_est = np.log(rmad) if (rmad := mad(resp)) > 0 else np.log(1e-16)

    # Fit the Model
    # Starting parameters for the Model
    a0 = mmed  # Use largest response with desired directionality
    if a0 == 0:
        a0 = 0.01  # If 0, use a smallish number
    guess = [a0 / 2, conc_max, er_est]  # y scale (a), x scale (b), logSigma (er)

    # Generate the bound matrices to constrain the model
    Ui = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
        ]
    )

    if not bidirectional:
        bnds = [0, -1e8 * np.abs(a0), 1e-8 * conc_max, -1e8 * conc_max]
    else:
        bnds = [-1e8 * np.abs(a0), -1e8 * np.abs(a0), 1e-8 * conc_max, -1e8 * conc_max]

    Ci = np.array(bnds).reshape(-1, 1)

    # Optimize the model
    
    fname = "poly2"
    func = globals()[fname]
    args = (conc, resp, func)
    result = minimize(tcplObj, x0=guess, args=args)
    # Generate some summary statistics
    if result.success:  # The model fit the data
        if verbose:
            print("poly2 >>>", result.nfev, result.success)

        success = 1
        aic = 2 * len(result.x) - 2 * result.fun
        for i, param in enumerate(pars):
            fenv[param] = result.x[i]

        # Calculate rmse for gnls
        modl = poly2(result.x, conc)
        rme = np.sqrt(np.mean((modl - resp) ** 2, axis=None, keepdims=False))

        # Calculate the sd for the gnls parameters
        try:
            hess_inv = np.linalg.inv(np.gradient(np.gradient(result.hess)))
            cov = 1
            diag_sqrt = np.sqrt(np.diag(hess_inv))
            if np.any(np.isnan(diag_sqrt)):
                for param_sd in sds:
                    fenv[param_sd] = np.nan
            else:
                for i, param_sd in enumerate(sds):
                    fenv[param_sd] = diag_sqrt[i]
        except np.linalg.LinAlgError:
            cov = 0
            for param_sd in sds:
                fenv[param_sd] = np.nan

    else:  # Curve did not fit the data
        success = 0
        aic = np.nan
        cov = np.nan
        rme = np.nan
        modl = np.nan
        sds = {param: np.nan for param in pars + sds}

    out = {
        "success": success,
        "aic": aic,
        "cov": cov,
        "rme": rme,
        "modl": modl,
        "pars": pars,
        "sds": sds,
    }

    return out