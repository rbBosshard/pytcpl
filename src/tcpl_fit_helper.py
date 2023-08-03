import numpy as np
from scipy.optimize import minimize

from acy import acy
from fit_models import get_fit_model
from tcpl_obj_fn import tcpl_obj


def fit_curve(fit_model, conc, resp, out):
    x = 10000
    a = 0.01
    b = 0.001
    c = 10

    initial_values = {
        'cnst': [1],
        'poly1': [1],
        'poly2': [20, 70],
        'pow': [1, 1],
        'exp4': [60, 10],
        'exp5': [60, 10, 2],
        'hill': [60, 10, 2],
        'gnls': [60, 10, 2, 60, 5],
    }

    bounds = {
        'cnst': ((a, x),),
        'poly1': ((-10, 10),),
        'poly2': ((-10, x), (-x, x)),
        'pow': ((-1000, 1000), (0.3, c)),
        'exp4': ((a, x), (b, x)),
        'exp5': ((a, x), (b, x), (a, c)),
        'hill': ((a, x), (b, x), (a, c)),
        'gnls': ((a, x), (b, x), (a, c), (b, x), (a, 20)),
    }

    initial_values = initial_values[fit_model] + [0.1]
    bounds = bounds[fit_model] + ((-1, 1),)
    args = (conc, resp, get_fit_model(fit_model))
    fit = minimize(tcpl_obj, x0=initial_values, bounds=bounds, args=args)

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

    # csv["cov"] = 0
    # try:
    #     # Estimate the covariance matrix using the inverse of the Hessian
    #     # Inverse of the objective function’s Hessian; may be an approximation. Not available for all solvers.
    #     covariance_matrix = np.linalg.inv(fit.hess_inv.todense())
    #     # uncertainties = standard deviations of parameters = diag_sqrt of covariance matrix
    #     uncertainties = np.sqrt(np.diag(covariance_matrix))

    #     if not np.any(np.isnan(uncertainties)):
    #         csv["cov"] = 1
    #         csv["sds"] = {param: uncertainties[i] for i, param in enumerate(csv["sds"])}
    #         # use taylor's theorem to approx sd's in change of units, only valid when sd's are << than ln(10)
    #         if fit_model == "hill":
    #             csv["sds"]["ga_sd"] = csv["pars"]["ga"] * np.log(10) * csv["sds"]["ga_sd"]
    #         if fit_model == "gnls":
    #             csv["sds"]["ga_sd"] = csv["pars"]["ga"] * np.log(10) * csv["sds"]["ga_sd"]
    #             csv["sds"]["la_sd"] = csv["pars"]["la"] * np.log(10) * csv["sds"]["la_sd"]

    # except Exception as e:
    #     print(f"{fit_model} >>> Error calculating parameter covariance: {e}")
    #     csv["cov"] = 0


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
            # csv["ac50_loss"] = None
        else:
            out["ac50"] = acy(.5 * out["top"], out, fit_model=fit_model)
            # csv["ac50_loss"] = acy(.5 * csv["top"], csv, fit_model=fit_model, getloss=True)

    return out


BMAD_CONSTANT = 1.4826


def mad(x):
    """Calculate the median absolute deviation (MAD) of an array"""
    return BMAD_CONSTANT * np.median(np.abs(x - np.median(x)))
