import numpy as np
from scipy.optimize import minimize
from pytcpl.acy import acy, tcpl_obj
from pytcpl.get_params import get_params
from pytcpl.get_fit_method import get_fit_method


def curve_fit(fit_method, conc, resp, bidirectional, to_fit, verbose):
    params = get_params(fit_method)

    # Prepare (nested) output dictionary
    out = {"pars": {p: None for p in params}, "sds": {p + "_sd": None for p in params},
           **{p: None for p in ["success", "aic", "cov", "rme", "modl"]}}

    if to_fit:
        initial_values, bounds = get_bounds_and_initial_values(fit_method, conc, resp, bidirectional, verbose)
        if fit_method in ["hill", "gnls"]:
            conc = np.log10(conc)
        args = (conc, resp, get_fit_method(fit_method), "basic")

        try:
            fit = minimize(tcpl_obj, x0=initial_values, bounds=bounds,  # method='L-BFGS-B',
                           args=args)  # bounds=bounds, method='L-BFGS-B', options={'jac': None, 'hess': None}?
            if verbose:
                print(f"{fit_method} >> success: {fit.success} {fit.message}, "
                      f"iter: {fit.nit}, evals: ({fit.nfev},{fit.njev})")
            try:
                out = generate_output(fit_method, conc, resp, out, fit, verbose)
            except Exception as e:
                print(f"{fit_method} >>> Error during generating output: {e}")
        except Exception as e:
            print(f"{fit_method} >>> Error during optimization: {e}")
            # fit = None
    return out


def get_bounds_and_initial_values(fit_method, conc, resp, bidirectional, verbose=False):
    unique_conc = np.unique(conc)
    logc = None
    conc_ = conc
    if fit_method in ["hill", "gnls"]:
        logc = np.log10(conc)
        unique_conc = np.unique(logc)
        conc_ = logc

    # get max response (i.e. max median response for multi-valued responses) and corresponding conc

    rmds = np.array([np.median(resp[conc_ == c]) for c in unique_conc])
    max_idx = np.argmax(np.abs(rmds)) if bidirectional else np.argmax(rmds)
    mmed = rmds[max_idx]
    mmed_conc = unique_conc[max_idx]

    # estimate error
    er_est = np.log(rmad) if (rmad := mad(resp)) > 0 else np.log(1e-16)

    conc_min = np.min(conc)
    conc_max = np.max(conc)

    # use largest response with desired directionality, if 0, use a smallish number
    a0 = mmed / conc_max if mmed != 0 else 0.01
    abs_a0 = abs(a0)
    lim_large = 1e8
    lim_small = 1e-8
    initial_values = []
    # Todo: extend for bidirectional == False
    # Todo: parameterize bounds variable, check for good values
    bounds = ()  # Assume bidirectional is True.
    if fit_method == "poly1":
        initial_values += [a0]
        bounds = ((-lim_large * abs_a0, lim_large * abs_a0),)
    if fit_method == "poly2":
        initial_values += [a0 / 2, conc_max]
        bounds = ((-lim_large * abs_a0, lim_large * abs_a0), (-lim_small * conc_max, lim_large * conc_max))
    if fit_method == "pow":
        initial_values += [a0, 1.5]
        bounds = ((-lim_large * abs_a0, lim_large * abs_a0), (1.01, 20))
    if fit_method == "exp2":
        initial_values += [a0, conc_max]
        bounds = ((lim_small * abs_a0, lim_large * abs_a0), (1e-2 * conc_max, lim_large * conc_max))
    if fit_method == "exp3":
        initial_values += [a0, conc_max, 1.2]
        bounds = ((lim_small * abs_a0, lim_large * abs_a0), (1e-2 * conc_max, lim_large * conc_max), (0.3, 8))
    if fit_method == "exp4":
        initial_values += [a0, mmed_conc/np.sqrt(10)]
        bounds = ((-1.2 * abs_a0, 1.2 * abs_a0), (conc_min/10, conc_max * np.sqrt(10)))
    if fit_method == "exp5":
        initial_values += [a0, mmed_conc/np.sqrt(10), 1.2]
        bounds = ((-1.2 * abs_a0, 1.2 * abs_a0), (conc_min/10, conc_max * np.sqrt(10)), (0.3, 8))
    if fit_method in ["hill", "gnls"]:
        logc_min = np.min(logc)
        logc_max = np.max(logc)
        resp_min = np.min(resp)
        resp_max = np.max(resp)
        val = 1.2 * max(np.abs(resp_min), np.abs(resp_max))
        initial_values += [mmed or 0.1, mmed_conc - 0.5, 1.2]
        bounds = ((-val, val), (logc_min - 1, logc_max + 0.5), (0.3, 8))
        if fit_method == "gnls":
            # Todo: constraint: la-ga >= minwidth = 1.5 is missing
            # minwidth = Minimum allowed distance between gain ac50 and loss ac50 (in log10 units)
            initial_values += [mmed_conc - 0.5 + 1.5 + 0.01, 5]
            bounds += ((logc_min - 1, logc_max + 2), (0.3, 8))

    # last step always append 1) er_est to initial_values, and 2) (None, None) to bounds
    initial_values += [er_est]
    bounds += ((None, None),)
    return np.array(initial_values), bounds


def generate_output(fit_method, conc, resp, out, fit, verbose=False):
    out["success"] = fit.success
    if not fit.success:
        pass  # Set breakpoint here
    out["aic"] = 2 * len(fit.x) + 2 * fit.fun
    out["pars"] = {param: fit.x[i] for i, param in enumerate(out["pars"])}
    out["modl"] = get_fit_method(fit_method)(fit.x, conc)
    out["rme"] = np.sqrt(np.mean((out["modl"] - resp) ** 2))
    assign_extra_attributes(fit_method, out)
    out["cov"] = 0
    # try:
    #     # Estimate the covariance matrix using the inverse of the Hessian
    #     # Inverse of the objective function’s Hessian; may be an approximation. Not available for all solvers.
    #     covariance_matrix = np.linalg.inv(fit.hess_inv.todense())
    #     # Access the uncertainty estimates
    #     uncertainties = np.sqrt(np.diag(
    #         covariance_matrix))  # uncertainties = standard deviations of parameters = diag_sqrt of covariance matrix
    #
    #     if not np.any(np.isnan(uncertainties)):
    #         out["cov"] = 1
    #         out["sds"] = {param: uncertainties[i] for i, param in enumerate(out["sds"])}
    #         # use taylor's theorem to approximate sd's in change of units
    #         # (only valid when sd's are much smaller than ln(10))
    #         if fit_method == "hill":
    #             out["sds"]["ga_sd"] = out["pars"]["ga"] * np.log2(10) * out["sds"]["ga_sd"]
    #         if fit_method == "gnls":
    #             out["sds"]["ga_sd"] = out["pars"]["ga"] * np.log2(10) * out["sds"]["ga_sd"]
    #             out["sds"]["la_sd"] = out["pars"]["la"] * np.log2(10) * out["sds"]["la_sd"]
    #
    # except Exception as e:
    #     print(f"{fit_method} >>> Error calculating parameter covariance: {e}")
    #     out["cov"] = 0

    return out


def assign_extra_attributes(fit_method, out):
    if fit_method == "hill":
        out["pars"]["ga"] = 10**out["pars"]["ga"]
    if fit_method == "gnls":
        out["pars"]["ga"] = 10**out["pars"]["ga"]
        out["pars"]["la"] = 10**out["pars"]["la"]
    if fit_method in ("poly1", "poly2", "pow", "exp2", "exp3"):
        out["top"] = out["modl"][np.argmax(np.abs(out["modl"]))]  # top is taken to be highest model value
        out["ac50"] = acy(.5 * out["top"], out, fit_method=fit_method)
    elif fit_method in ("hill", "exp4", "exp5"):
        # methods with a theoretical top/ac50
        out["top"] = out["pars"]["tp"]
        out["ac50"] = out["pars"]["ga"]
    elif fit_method == "gnls":
        # gnls methods; use calculated top/ac50, etc.
        out["top"] = acy(0, out, fit_method=fit_method, returntop=True)
        # check if the theoretical top was calculated
        if np.isnan(out["top"]):
            # if the theoretical top is NA return NA for ac50 and ac50_loss
            out["ac50"] = np.nan
            out["ac50_loss"] = np.nan
        else:
            out["ac50"] = acy(.5 * out["top"], out, fit_method=fit_method)
            out["ac50_loss"] = acy(.5 * out["top"], out, fit_method=fit_method, getloss=True)
    return out


BMAD_CONSTANT = 1.4826


def mad(x):
    """Calculate the median absolute deviation (MAD) of an array"""
    return BMAD_CONSTANT * np.median(np.abs(x - np.median(x)))
