import numpy as np
from scipy.stats import t, norm
from concurrent.futures import ThreadPoolExecutor

from fit_method_helper import init_fit_method, fit
from tcplfit2_core_helper import prepare_tcplfit2_core, assign_extra_attributes

def tcplfit2_core(conc, resp, cutoff, fitmodels, bidirectional=False, verbose=False, force_fit=False):
    conc, resp, cutoff, rmds, fit_funcs = prepare_tcplfit2_core(conc, resp, cutoff, fitmodels)
    out = {}
    with ThreadPoolExecutor() as executor:
        futures = []
        for model in fitmodels:
            to_fit = get_to_fit_condition(cutoff, fitmodels, force_fit, rmds, model)
            futures.append(executor.submit(fit_funcs[model], conc, resp, bidirectional, to_fit))

        for future, model in zip(futures, fitmodels):
            model_results, to_fit = future.result()
            out[model] = assign_extra_attributes(model, to_fit, model_results)

    return out

def get_to_fit_condition(cutoff, fitmodels, force_fit, rmds, model):
    return len(rmds) >= 4 and model in fitmodels and (np.any(np.abs(rmds) >= cutoff) or force_fit or model == "cnst")


def tcplObj(ps, conc, resp, fname, errfun="dt4"):
    mu = fname(ps=ps, x=conc)  # get model values for each conc, ps = parameter vector
    err = np.exp(ps[-1]) # last parameter is the log of the error variance
    # objective function is the sum of log-likelihood of response given the model at each concentration scaled by variance (err) -> normalize residulas (take into consideration the relative importance of each residual in the objective function)
    # Wrapped with scipy.optimize.minimize(): It is the convention that we call the optimization objective function a "cost function" or "loss function" and therefore, we want to minimize them, rather than maximize them, and hence the negative log likelihood is formed, rather than positive likelihood in your word 
    # https://stats.stackexchange.com/questions/260505/why-do-we-use-negative-log-likelihood-to-estimate-parameters-for-ml
    if errfun == "dt4":
        nll =  -np.sum(t.logpdf((resp - mu) / err, df=4) - np.log(err))  # degree of freedom paramter = 4, for Student’s t probability density function
    elif errfun == "dnorm":
        nll -np.sum(norm.logpdf((resp - mu) / err) - np.log(err))
    
    return nll # negative log likelihood scaled by variance


def cnst(ps, x):
    # ignores ps
    return np.zeros(len(x))

def exp2(ps, x):
    # a = ps[0], b = ps[1]
    return ps[0] * (np.exp(x / ps[1]) - 1)

def exp3(ps, x):
    # a = ps[0], b = ps[1], p = ps[2]
    return ps[0] * (np.exp((x / ps[1]) ** ps[2]) - 1)

def exp4(ps, x):
    # tp = ps[0], ga = ps[1]
    return ps[0] * (1 - 2 ** (-x / ps[1]))

def exp5(ps, x):
    # tp = ps[0], ga = ps[1], p = ps[2]
    return ps[0] * (1 - 2 ** (-(x / ps[1]) ** ps[2]))

def gnls(ps, x):
    # gnls function with regular units
    # tp = ps[0], ga = ps[1], p = ps[2], la = ps[3], q = ps[4]
    gn = 1 / (1 + (ps[1] / x) ** ps[2])
    ls = 1 / (1 + (x / ps[3]) ** ps[4])
    return ps[0] * gn * ls

def loggnls(ps, x):
    # gnls function with log units: x = log10(conc) and ga/la = log10(gain/loss ac50)
    # tp = ps[0], ga = ps[1], p = ps[2], la = ps[3], q = ps[4]
    gn = 1 / (1 + 10 ** ((ps[1] - x) * ps[2]))
    ls = 1 / (1 + 10 ** ((x - ps[3]) * ps[4]))
    return ps[0] * gn * ls

def hillfn(ps, x):
    # hill function with regular units
    # tp = ps[0], ga = ps[1], p = ps[2]
    return ps[0] / (1 + (ps[1] / x) ** ps[2])

def loghill(ps, x):
    # hill function with log units: x = log10(conc) and ga = log10(ac50)
    # tp = ps[0], ga = ps[1], p = ps[2]
    return ps[0] / (1 + 10 ** (ps[2] * (ps[1] - x)))

def poly1(ps, x):
    # a = ps[0]
    return ps[0] * x

def poly2(ps, x):
    # a = ps[0], b = ps[1]
    x0 = x / ps[1]
    return ps[0] * (x0 + x0 * x0)

def pow(ps, x):
    # a = ps[0], p = ps[1]
    return ps[0] * x ** ps[1]