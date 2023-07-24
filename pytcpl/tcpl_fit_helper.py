import numpy as np
from scipy.optimize import minimize, curve_fit, Bounds

from acy import acy
from fit_models import get_fit_model
from tcpl_obj_fn import tcpl_obj

def fit_curve(fit_model, conc, resp, bidirectional, out, fit_strategy):
    initial_values, bounds, linear_constraints = get_bounds_and_initial_values(fit_model, fit_strategy, conc, resp,
                                                                               bidirectional)
    try:
        if fit_strategy == "mle":
            args = (conc, resp, get_fit_model(fit_model), fit_strategy)
            fit = minimize(tcpl_obj, x0=np.array(initial_values), bounds=bounds, constraints=linear_constraints, args=args)
            popt = fit.x
            # print(f"{fit_model} >> success: {fit.success} {fit.message}, "
            #       f"iter: {fit.nit}, evals: ({fit.nfev},{fit.njev})")
        elif fit_strategy == "leastsq":
            result = curve_fit(get_fit_model(fit_model), xdata=conc, ydata=resp, p0=initial_values, bounds=bounds, full_output=True)
            popt, pcov, infodict = result[0], result[1][0], result[2]
        else:
            raise NotImplementedError("Fit strategy not supported.")

        generate_output(fit_model, conc, resp, out, popt, fit_strategy)

    except Exception as e:
        print(f"{fit_model} >>> Error during optimization or generating output: {e}")
        

def get_bounds_and_initial_values(fit_model, fit_strategy, conc, resp, bidirectional):
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
    # Todo: extend for bidirectional == False
    # Todo: set correct bounds/constraints
    initial_values = []
    lbs = []
    ubs = []
    linear_constraints = None
    if fit_model == "poly1":
        # initial_values.append(a0)
        initial_values.append(0.79200809)
        lbs.append(-lim_large * abs_a0)
        ubs.append(lim_large * abs_a0)
    if fit_model == "poly2":
        # initial_values.append(a0 / 2)
        initial_values.append(3.39713788e+06)
        lbs.append(-lim_large * abs_a0)
        ubs.append(lim_large * abs_a0)
        # initial_values.append(conc_max)
        initial_values.append(7.74877015e+08)
        lbs.append(lim_small * conc_max)
        ubs.append(lim_large * conc_max)
    if fit_model == "pow":
        # initial_values.append(a0)
        initial_values.append(1.2165843)
        lbs.append(-lim_large * abs_a0)
        ubs.append(lim_large * abs_a0)
        # initial_values.append(1.5)
        initial_values.append(0.8229419)
        lbs.append(0.3)
        ubs.append(20)
    if fit_model == "exp2":
        # initial_values.append(a0)
        initial_values.append(1371.56420724)
        lbs.append(-lim_large * abs_a0)
        ubs.append(lim_large * abs_a0)
        # initial_values.append(conc_max)
        initial_values.append(797582.77160091)
        lbs.append(1e-2 * conc_max)
        ubs.append(lim_large * conc_max)
    if fit_model == "exp3":
        initial_values.append(a0)
        # initial_values.append(4.28780012e+02)
        lbs.append(-lim_large * abs_a0)
        ubs.append(lim_large * abs_a0)
        initial_values.append(conc_max)
        # initial_values.append(9.13083436e+09)
        lbs.append(1e-2 * conc_max)
        ubs.append(lim_large * conc_max)
        initial_values.append(1.2)
        # initial_values.append(9.41513409e-01)
        lbs.append(0.3)
        ubs.append(8)
    if fit_model == "exp4":
        initial_values.append(a0)
        # initial_values.append(0.02560154)
        lbs.append(-1.2 * abs_a0)
        ubs.append(1.2 * abs_a0)
        initial_values.append(mmed_conc/np.sqrt(10))
        # initial_values.append(0.20462039)
        lbs.append(conc_min/10)
        ubs.append(conc_max * np.sqrt(10))
    if fit_model == "exp5":
        initial_values.append(a0)
        # initial_values.append(0.19415434)
        lbs.append(-1.2 * abs_a0)
        ubs.append(1.2 * abs_a0)
        initial_values.append(mmed_conc/np.sqrt(10))
        # initial_values.append(19.60645484)
        lbs.append(conc_min/10)
        ubs.append(conc_max * np.sqrt(10))
        initial_values.append(1.2)
        # initial_values.append(3.53210695)
        lbs.append(0.3)
        ubs.append(8)
    if fit_model in ["hill", "gnls"]:
        resp_min = np.min(resp)
        resp_max = np.max(resp)
        val = 1.2 * max(np.abs(resp_min), np.abs(resp_max))
        initial_values.append(mmed or 0.1)
        # initial_values.append(0.21679661)
        lbs.append(-val)
        ubs.append(val)
        initial_values.append(conc_min/5)
        # initial_values.append(21.89003274)
        lbs.append(conc_min/10)
        ubs.append(conc_max * 5)
        initial_values.append(1.2)
        # initial_values.append(5.67061821)
        lbs.append(0.3)
        ubs.append(8)
        if fit_model == "gnls":
            initial_values.append(mmed_conc * 10.1)
            # initial_values.append(1.19582880e+03)
            lbs.append(conc_min/10)
            ubs.append(conc_max * 20)
            initial_values.append(5)
            # initial_values.append(54.74798835)
            lbs.append(0.3)
            ubs.append(8)
            # constraint: la-ga >= minwidth=1.5, (in log10 units) min allowed dist between gain.ac50 & loss.ac50
            # https://towardsdatascience.com/introduction-to-optimization-constraints-with-scipy-7abd44f6de25#a9d0
            # linear_constraints = LinearConstraint([[0, -1, 0, 1, 0, 0]], [10 ** 1.5], [np.inf])

    # For last param "err": append er_est to initial_values, and (None, None) to bounds
    if fit_strategy == "mle":
        initial_values.append(er_est)
        lbs.append(-lim_large)
        ubs.append(lim_large)

    bounds = Bounds(lbs, ubs)
    return initial_values, bounds, linear_constraints


def generate_output(fit_model, conc, resp, out, fit, fit_strategy):
    # print(f"{fit_model} > iter: {fit.nit}, evals: ({fit.nfev},{fit.njev})")
    out["success"] = 1
    # Compute the fitted values
    params = fit if fit_strategy == "leastsq" else fit[:-1]
    y_fit = get_fit_model(fit_model)(conc, *params).tolist()
    # Calculate the sum of squared residuals (SSR)
    mse = np.square(np.subtract(resp, y_fit)).mean()
    # Compute the number of model parameters (k)
    num_parameters = len(fit) - 1 if fit_strategy == "leastsq" else len(fit)
    out["aic"] = 2 * num_parameters + len(conc) * np.log(mse)
    out["pars"] = {param: fit[i] for i, param in enumerate(out["pars"])}
    out["modl"] = y_fit
    # out["rme"] = np.sqrt(mse)
    assign_extra_attributes(fit_model, out)
    del out["modl"]
    # out["cov"] = 0
    # try:
    #     # Estimate the covariance matrix using the inverse of the Hessian
    #     # Inverse of the objective function’s Hessian; may be an approximation. Not available for all solvers.
    #     covariance_matrix = np.linalg.inv(fit.hess_inv.todense())
    #     # uncertainties = standard deviations of parameters = diag_sqrt of covariance matrix
    #     uncertainties = np.sqrt(np.diag(covariance_matrix))

    #     if not np.any(np.isnan(uncertainties)):
    #         out["cov"] = 1
    #         out["sds"] = {param: uncertainties[i] for i, param in enumerate(out["sds"])}
    #         # use taylor's theorem to approx sd's in change of units, only valid when sd's are << than ln(10)
    #         if fit_model == "hill":
    #             out["sds"]["ga_sd"] = out["pars"]["ga"] * np.log(10) * out["sds"]["ga_sd"]
    #         if fit_model == "gnls":
    #             out["sds"]["ga_sd"] = out["pars"]["ga"] * np.log(10) * out["sds"]["ga_sd"]
    #             out["sds"]["la_sd"] = out["pars"]["la"] * np.log(10) * out["sds"]["la_sd"]

    # except Exception as e:
    #     print(f"{fit_model} >>> Error calculating parameter covariance: {e}")
    #     out["cov"] = 0



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
