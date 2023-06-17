import json

import numpy as np
from scipy.optimize import minimize, LinearConstraint
from acy import acy
from tcpl_obj_fn import tcpl_obj
from fit_models import get_params
from fit_models import get_fit_model
import autograd
from numdifftools import Jacobian, Hessian


def fit_curve(fit_model, conc, resp, bidirectional, out, verbose):

    initial_values, bounds, linear_constraints = get_bounds_and_initial_values(fit_model, conc, resp, bidirectional, verbose)

    args = (conc, resp, get_fit_model(fit_model))

    try:
        # bounds=bounds, method='L-BFGS-B', constraints=linear_constraints,
        fit = minimize(tcpl_obj, x0=initial_values, bounds=bounds, constraints=linear_constraints, args=args)

        if verbose > 1:
            print(f"{fit_model} >> success: {fit.success} {fit.message}, "
                  f"iter: {fit.nit}, evals: ({fit.nfev},{fit.njev})")

        generate_output(fit_model, conc, resp, out, fit, verbose)

    except Exception as e:
        print(f"{fit_model} >>> Error during optimization or generating output: {e}")



def get_bounds_and_initial_values(fit_model, conc, resp, bidirectional, verbose=False):
    unique_conc = np.unique(conc)

    # get max response (i.e. max median response for multi-valued responses) and corresponding conc
    rmds = np.array([np.median(resp[conc == c]) for c in unique_conc])
    max_idx = np.argmax(np.abs(rmds)) if bidirectional else np.argmax(rmds)
    mmed = rmds[max_idx]
    mmed_conc = unique_conc[max_idx]

    # estimate error
    er_est = np.log(rmad) if (rmad := mad(resp)) > 0 else np.log(1e-16)

    conc_min = np.min(conc)
    conc_max = np.max(conc)

    # use largest response with desired directionality, if 0, use a smallish number
    a0 = mmed or 0.01
    abs_a0 = abs(a0)
    lim_large = 1e8
    lim_small = 1e-8
    lim_small2 = 1e-2
    initial_values = []
    # Todo: extend for bidirectional == False
    # Todo: set correct bounds/constraints
    bounds = ()  # Assume bidirectional is True.
    linear_constraints = None
    if fit_model == "poly1":
        initial_values += [a0]
        bounds = ((-lim_large * abs_a0, lim_large * abs_a0),)
    if fit_model == "poly2":
        initial_values += [a0 / 2, conc_max]
        bounds = ((-lim_large * abs_a0, lim_large * abs_a0), (lim_small * conc_max, lim_large * conc_max))
    if fit_model == "pow":
        initial_values += [a0, 1.5]
        bounds = ((-lim_large * abs_a0, lim_large * abs_a0), (0.3, 20))
    if fit_model == "exp2":
        initial_values += [a0, conc_max]
        bounds = ((-lim_large * abs_a0, lim_large * abs_a0), (1e-2 * conc_max, lim_large * conc_max))
    if fit_model == "exp3":
        initial_values += [a0, conc_max, 1.2]
        bounds = ((-lim_large * abs_a0, lim_large * abs_a0), (1e-2 * conc_max, lim_large * conc_max), (0.3, 8))
    if fit_model == "exp4":
        initial_values += [a0, mmed_conc/np.sqrt(10)]
        bounds = ((-1.2 * abs_a0, 1.2 * abs_a0), (conc_min/10, conc_max * np.sqrt(10)))
    if fit_model == "exp5":
        initial_values += [a0, mmed_conc/np.sqrt(10), 1.2]
        bounds = ((-1.2 * abs_a0, 1.2 * abs_a0), (conc_min/10, conc_max * np.sqrt(10)), (0.3, 8))
    if fit_model in ["hill", "gnls"]:
        resp_min = np.min(resp)
        resp_max = np.max(resp)
        val = 1.2 * max(np.abs(resp_min), np.abs(resp_max))
        initial_values += [mmed or 0.1, conc_min/5, 1.2]
        bounds = ((-val, val), (conc_min/10, conc_max * 5), (0.3, 8))
        if fit_model == "gnls":
            initial_values += [mmed_conc * 10.1, 5]
            bounds += ((conc_min/10, conc_max * 20), (0.3, 8))
            # constraint: la-ga >= minwidth=1.5, (in log10 units) min allowed dist between gain.ac50 & loss.ac50
            # https://towardsdatascience.com/introduction-to-optimization-constraints-with-scipy-7abd44f6de25#a9d0
            # linear_constraints = LinearConstraint([[0, -1, 0, 1, 0, 0]], [10 ** 1.5], [np.inf])

    # For last param "err": append er_est to initial_values, and (None, None) to bounds
    initial_values += [er_est]
    bounds += ((None, None),)
    return np.array(initial_values), bounds, linear_constraints


def generate_output(fit_model, conc, resp, out, fit, verbose=False):
    if not fit.success:
        # print(f"{fit_model} >> success: {fit.success} {fit.message}, "
        #       f"iter: {fit.nit}, evals: ({fit.nfev},{fit.njev})")
        pass  # Set breakpoint here
    else:
        # print(f"{fit_model} > iter: {fit.nit}, evals: ({fit.nfev},{fit.njev})")
        pass

    out["success"] = float(fit.success)
    out["aic"] = 2 * len(fit.x) + 2 * fit.fun
    out["pars"] = {param: fit.x[i] for i, param in enumerate(out["pars"])}
    out["modl"] = get_fit_model(fit_model)(fit.x, conc).tolist()
    out["rme"] = np.sqrt(np.mean((out["modl"] - resp) ** 2))
    assign_extra_attributes(fit_model, out)

    out["cov"] = 0
    try:
        # Estimate the covariance matrix using the inverse of the Hessian
        # Inverse of the objective function’s Hessian; may be an approximation. Not available for all solvers.
        covariance_matrix = np.linalg.inv(fit.hess_inv.todense())
        # uncertainties = standard deviations of parameters = diag_sqrt of covariance matrix
        uncertainties = np.sqrt(np.diag(covariance_matrix))

        if not np.any(np.isnan(uncertainties)):
            out["cov"] = 1
            out["sds"] = {param: uncertainties[i] for i, param in enumerate(out["sds"])}
            # use taylor's theorem to approx sd's in change of units, only valid when sd's are << than ln(10)
            if fit_model == "hill":
                out["sds"]["ga_sd"] = out["pars"]["ga"] * np.log(10) * out["sds"]["ga_sd"]
            if fit_model == "gnls":
                out["sds"]["ga_sd"] = out["pars"]["ga"] * np.log(10) * out["sds"]["ga_sd"]
                out["sds"]["la_sd"] = out["pars"]["la"] * np.log(10) * out["sds"]["la_sd"]

    except Exception as e:
        print(f"{fit_model} >>> Error calculating parameter covariance: {e}")
        out["cov"] = 0



def assign_extra_attributes(fit_model, out):
    if fit_model in ("poly1", "poly2", "pow", "exp2", "exp3"):
        out["top"] = out["modl"][np.argmax(np.abs(out["modl"]))]  # top is taken to be highest model value
        out["ac50"] = acy(.5 * out["top"], out, fit_model=fit_model)
    elif fit_model in ("hill", "exp4", "exp5"):
        # methods with a theoretical top/ac50
        out["top"] = out["pars"]["tp"]
        out["ac50"] = out["pars"]["ga"]
    elif fit_model == "gnls":
        # gnls methods; use calculated top/ac50, etc.
        out["top"] = acy(0, out, fit_model=fit_model, returntop=True)
        # check if the theoretical top was calculated
        if np.isnan(out["top"]):
            # if the theoretical top is NA return NA for ac50 and ac50_loss
            out["ac50"] = None
            out["ac50_loss"] = None
        else:
            out["ac50"] = acy(.5 * out["top"], out, fit_model=fit_model)
            out["ac50_loss"] = acy(.5 * out["top"], out, fit_model=fit_model, getloss=True)
    return out


BMAD_CONSTANT = 1.4826


def mad(x):
    """Calculate the median absolute deviation (MAD) of an array"""
    return BMAD_CONSTANT * np.median(np.abs(x - np.median(x)))
