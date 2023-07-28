import numpy as np
from scipy.optimize import minimize, curve_fit, Bounds

from acy import acy
from fit_models import get_fit_model
from tcpl_obj_fn import tcpl_obj


def fit_curve(fit_model, conc, resp, bidirectional, out, fit_strategy):
    initial_values, bounds, linear_constraints = get_bounds_and_initial_values(fit_model, fit_strategy, conc, resp,
                                                              bidirectional)

    x = 100.0
    y = 1
    z = 1
    er_ = None
    # Updated initial values and bounds with the "err" parameter
    initial_values = {
        'cnst': [0.9],
        'poly1': [0.9],
        'poly2': [20, 70],
        'pow': [0.9, 0.9],
        'exp2': [40, 70],
        'exp3': [50, 90, 0.9],
        'exp4': [60, 10],
        'exp5': [60, 10, 2],
        'hill': [60, 10, 2],
        'gnls': [60, 10, 2, 60, 5],
        'expo': [z, z],
    }

    bounds = {
        'cnst': ((-x, x),),
        'poly1': ((-x, x),),
        'poly2': ((-x, x), (-x, x)),
        'pow': ((-x, x), (0.3, 20)),
        'exp2': ((-x, x), (0.1, x)),
        'exp3': ((-x, x), (0.1, x), (0.3, 8)),
        'exp4': ((-x, x), (0.1, x)),
        'exp5': ((-x, x), (0.1, x), (0.3, 8)),
        'hill': ((-x, x), (0.1, x), (0.3, 8)),
        'gnls': ((-x, x), (0.1, x), (0.3, 8), (0.1, x), (0.3, 8)),
        'expo': ((0.1, x), (-x, x), (-x, x)),
    }

    def negative_log_likelihood(params, conc, response, model):
        predicted_response = model(conc, *params[:-1])
        error = response - predicted_response
        sigma_squared = np.var(error)
        n = len(conc)
        return n / 2.0 * np.log(2 * np.pi * sigma_squared) + 0.5 / sigma_squared * np.sum(error ** 2)

    initial_values = initial_values[fit_model] + [0.1]
    bounds = bounds[fit_model] + ((-1, 1),)
    args = (conc, resp, get_fit_model(fit_model))
    fit = minimize(tcpl_obj, x0=initial_values, bounds=bounds, args=args, )

    try:
        generate_output(fit_model, conc, resp, out, fit, fit_strategy)
    except Exception as e:
        print(f"{fit_model}: {e}")
    a = 1

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
    lim_small2 = 1e-2
    initial_values = []
    # Todo: extend for bidirectional == False
    # Todo: set correct bounds/constraints
    bounds = ()  # Assume bidirectional is True.
    linear_constraints = None
    if fit_model == "cnst":
        initial_values += [a0]
        bounds = ((-lim_large * abs_a0, lim_large * abs_a0),)
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
        initial_values += [a0, mmed_conc / np.sqrt(10)]
        bounds = ((-1.2 * abs_a0, 1.2 * abs_a0), (conc_min / 10, conc_max * np.sqrt(10)))
    if fit_model == "exp5":
        initial_values += [a0, mmed_conc / np.sqrt(10), 1.2]
        bounds = ((-1.2 * abs_a0, 1.2 * abs_a0), (conc_min / 10, conc_max * np.sqrt(10)), (0.3, 8))
    if fit_model in ["hill", "gnls"]:
        resp_min = np.min(resp)
        resp_max = np.max(resp)
        val = 1.2 * max(np.abs(resp_min), np.abs(resp_max))
        initial_values += [mmed or 0.1, conc_min / 5, 1.2]
        bounds = ((-val, val), (conc_min / 10, conc_max * 5), (0.3, 8))
        if fit_model == "gnls":
            initial_values += [mmed_conc, 0.8]
            bounds += ((conc_min / 10, conc_max * 20), (0.3, 8))
            # constraint: la-ga >= minwidth=1.5, (in log10 units) min allowed dist between gain.ac50 & loss.ac50
            # https://towardsdatascience.com/introduction-to-optimization-constraints-with-scipy-7abd44f6de25#a9d0
            # linear_constraints = LinearConstraint([[0, -1, 0, 1, 0, 0]], [10 ** 1.5], [np.inf])

    # For last param "err": append er_est to initial_values, and (None, None) to bounds
    initial_values += [er_est]
    bounds += ((None, None),)
    return np.array(initial_values), bounds, linear_constraints


def calculate_aic(nll, num_params):
    return 2 * num_params - 2 * nll


def generate_output(fit_model, conc, resp, out, fit, fit_strategy):
    fit_params = fit.x
    num_params = len(fit_params)
    log_likelihood = -fit.fun  # the output was the negative log-likelihood
    # print(f"{fit_model} > iter: {fit.nit}, evals: ({fit.nfev},{fit.njev})")
    pred = get_fit_model(fit_model)(conc, *fit_params[:-1]).tolist()
    # Let k be the number of estimated params anl L the max value of the likelihood function for the model. Then
    # AIC = 2 * k - 2 * ln(L)
    aic = 2 * num_params - 2 * log_likelihood
    out["aic"] = aic
    if aic < 0:
        exit()
    out["pars"] = {key: value for key, value in zip(out["pars"].keys(), fit_params)}
    out["modl"] = pred
    out["rmse"] = np.sqrt(np.mean((resp - pred) ** 2))
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
    if fit_model in ("poly1", "poly2", "pow", "exp2", "exp3", "expo"):
        out["top"] = out["modl"][np.argmax(np.abs(out["modl"]))]  # top is taken to be highest model value
        ac50 = acy(.5 * out["top"], out, fit_model=fit_model)
        out["ac50"] = ac50
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
            # out["ac50_loss"] = None
        else:
            out["ac50"] = acy(.5 * out["top"], out, fit_model=fit_model)
            # out["ac50_loss"] = acy(.5 * out["top"], out, fit_model=fit_model, getloss=True)

    return out


BMAD_CONSTANT = 1.4826


def mad(x):
    """Calculate the median absolute deviation (MAD) of an array"""
    return BMAD_CONSTANT * np.median(np.abs(x - np.median(x)))
