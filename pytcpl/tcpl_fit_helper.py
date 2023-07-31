import numpy as np
from scipy.optimize import minimize, curve_fit, Bounds

from acy import acy
from fit_models import get_fit_model
from tcpl_obj_fn import tcpl_obj


def fit_curve(fit_model, conc, resp, bidirectional, out):
    x = 100.0
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
        'expo': [1, 1],
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
        'expo': ((0.1, x), (-x, x)),
    }

    initial_values = initial_values[fit_model] + [0.1]
    bounds = bounds[fit_model] + ((-1, 1),)
    args = (conc, resp, get_fit_model(fit_model))
    fit = minimize(tcpl_obj, x0=initial_values, bounds=bounds, args=args, )

    try:
        generate_output(fit_model, conc, resp, out, fit)
    except Exception as e:
        print(f"{fit_model}: {e}")


def generate_output(fit_model, conc, resp, out, fit):
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
