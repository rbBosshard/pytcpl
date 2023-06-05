import numpy as np
from mad import mad
from scipy.optimize import minimize
from pytcpl.acy import acy, tcplObj


def curve_fit(fitmethod, conc, resp, bidirectional, to_fit):
    params = {
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
    }.get(fitmethod)

    # Prepare (nested) output dictionary
    out = {"pars": {p: None for p in params}, "sds": {p + "_sd": None for p in params},
           **{p: None for p in ["success", "aic", "cov", "rme", "modl"]}}

    if to_fit:
        initial_values, bounds = get_bounds_and_initial_values(fitmethod, conc, resp, bidirectional)
        args = (conc, resp, globals()[fitmethod])

        try:
            fit = minimize(tcplObj, x0=initial_values, method='L-BFGS-B', args=args)  # bounds=bounds,
            out = generate_output(fitmethod, conc, resp, out, fit)
        except Exception as e:
            print(f"{fitmethod} >>> Error during optimization: {e}")

    return out


def get_bounds_and_initial_values(fitmethod, conc, resp, bidirectional):
    # median at each conc, for multi-valued responses
    unique_conc = np.unique(conc)
    rmds = np.array([np.median(resp[conc == c]) for c in unique_conc])
    # get max response and corresponding conc
    mmed = rmds[np.argmax(rmds)] if not bidirectional else rmds[np.argmax(np.abs(rmds))]
    er_est = np.log(rmad) if (rmad := mad(resp)) > 0 else np.log(1e-16)
    conc_max = np.max(conc)
    # use largest response with desired directionality, if 0, use a smallish number
    a0 = mmed / conc_max if mmed != 0 else 0.01
    abs_a0 = abs(a0)
    lim = 1e-8
    initial_values = []
    bounds = ()  # Assume bidirectional is True!
    if fitmethod == "poly1":
        initial_values += [a0]
        bounds += ((-lim * abs_a0, lim * abs_a0),)
    if fitmethod == "poly2":
        initial_values += [a0 / 2, conc_max]
        bounds += ((-lim * abs_a0, lim * abs_a0), (-lim * conc_max, lim * conc_max))

    # last step always append 1) er_est to initial_values, and 2) (None, None) to bounds
    initial_values += [er_est]
    bounds += ((None, None),)
    return np.array(initial_values), bounds


def generate_output(fitmethod, conc, resp, out, fit):
    out["success"] = 1
    out["aic"] = 2 * len(fit.x) + 2 * fit.fun
    out["pars"] = {param: fit.x[i] for i, param in enumerate(out["pars"])}
    out["modl"] = globals()[fitmethod](fit.x, conc)
    out["rme"] = np.sqrt(np.mean((out["modl"] - resp) ** 2))
    assign_extra_attributes(fitmethod, out)
    try:
        # Estimate the covariance matrix using the inverse of the Hessian
        # Inverse of the objective function’s Hessian; may be an approximation. Not available for all solvers.
        covariance_matrix = np.linalg.inv(fit.hess_inv.todense())
        # Access the uncertainty estimates
        uncertainties = np.sqrt(np.diag(
            covariance_matrix))  # uncertainties = standard deviations of parameters = diag_sqrt of covariance matrix

        if not np.any(np.isnan(uncertainties)):
            out["cov"] = 1
            out["sds"] = {param: uncertainties[i] for i, param in enumerate(out["sds"])}

    except Exception as e:
        print(f"{fitmethod} >>> Error calculating parameter covariance: {e}")
        out["cov"] = 0

    return out


def assign_extra_attributes(fitmethod, out):
    if fitmethod in ("poly1", "poly2", "pow", "exp2", "exp3"):
        out["top"] = out["modl"][np.argmax(np.abs(out["modl"]))]  # top is taken to be highest model value
        out["ac50"] = acy(.5 * out["top"], out, type=fitmethod)
    elif fitmethod in ("hill", "exp4", "exp5"):
        # methods with a theoretical top/ac50
        out["top"] = out["tp"]
        out["ac50"] = out["ga"]
    elif fitmethod == "gnls":
        # gnls methods; use calculated top/ac50, etc.
        out["top"] = acy(0, out, type=fitmethod, returntop=True)
        # check if the theoretical top was calculated
        if np.isnan(out["top"]):
            # if the theoretical top is NA return NA for ac50 and ac50_loss
            out["ac50"] = np.nan
            out["ac50_loss"] = np.nan
        else:
            out["ac50"] = acy(.5 * out["top"], out, type=fitmethod)
            out["ac50_loss"] = acy(.5 * out["top"], out, type=fitmethod, getloss=True)
    return out
