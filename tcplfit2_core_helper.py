import numpy as np
from acy import acy

from fit_method_helper import init_fit_method, fit

def prepare_tcplfit2_core(conc, resp, cutoff, fitmodels):
    # conc = np.array(conc.iloc[0])
    # resp = np.array(resp.iloc[0])
    # cutoff = np.array(cutoff.iloc[0])
    logc = np.log10(conc)
    # median at each conc, for multi-valued responses
    unique_conc = np.unique(logc)
    rmds = np.array([np.median(resp[logc == c]) for c in unique_conc])

    # Check for edge case where all responses are equal
    if np.max(resp) == np.min(resp) and resp[0] == 0: # check if all response values are zero
        print("all response values are 0: add epsilon (1e-6) to all response elements.")
        print("Response Range:", np.min(resp), np.max(resp))
        resp += 1e-6  # adding epsilon to resp vector

    fit_funcs = {}
    for model in fitmodels:
        fit_funcs[model] = eval("fit" + model)
    return conc,resp,cutoff,rmds,fit_funcs

def assign_extra_attributes(model, to_fit, model_results):
    if to_fit:
        if model in ("poly1", "poly2", "pow", "exp2", "exp3"):
            model_results["top"] = model_results["modl"][np.argmax(np.abs(model_results["modl"]))] # top is taken to be highest model value
            model_results["ac50"] = acy(.5 * model_results["top"], model_results, type=model)
        elif model in ("hill", "exp4", "exp5"):
            # methods with a theoretical top/ac50
            model_results["top"] = model_results["tp"]
            model_results["ac50"] = model_results["ga"]
        elif model == "gnls":
            # gnls methods; use calculated top/ac50, etc.
            model_results["top"] = acy(0, model_results, type=model, returntop=True)
            # check if the theoretical top was calculated
            if np.isnan(model_results["top"]):
                # if the theoretical top is NA return NA for ac50 and ac50_loss
                model_results["ac50"] = np.nan
                model_results["ac50_loss"] = np.nan
            else:
                model_results["ac50"] = acy(.5 * model_results["top"], model_results, type=model)
                model_results["ac50_loss"] = acy(.5 * model_results["top"], model_results, type=model, getloss=True)
    return model_results




def fitcnst(conc, resp, bidirectional=True, to_fit=False):
    fitmethod = "cnst"
    pars, sds, out, mmed, er_est, args = init_fit_method(fitmethod, conc, resp)
    
    if to_fit:
        guess = [er_est]  # linear coeff (a); set to run through the max resp at the max conc
        out = fit(conc, resp, fitmethod, pars, sds, out, args, guess)
    
    return out, to_fit


def fitpoly1(conc, resp, bidirectional=True, to_fit=False):
    fitmethod = "poly1"
    pars, sds, out, mmed, er_est, args = init_fit_method(fitmethod, conc, resp, bidirectional)
      
    if to_fit:
        conc_max = np.max(conc)
    
        # Starting parameters for the Model
        a0 = mmed / conc_max  # use largest response with desired directionality
        if a0 == 0:
            a0 = 0.01  # if 0, use a smallish number
        guess = [a0, er_est]  # linear coeff (a); set to run through the max resp at the max conc

        out = fit(conc, resp, fitmethod, pars, sds, out, args, guess)
    
    return out, to_fit



def fitpoly2(conc, resp, bidirectional=True, to_fit=False):
    fitmethod = "poly2"
    pars, sds, out, mmed, er_est, args = init_fit_method(fitmethod, conc, resp, bidirectional)
      
    if to_fit:
        conc_max = np.max(conc)
    
        # Starting parameters for the Model
        a0 = mmed / conc_max 
        if a0 == 0:
            a0 = 0.01
        guess = [a0 / 2, conc_max, er_est]

        out = fit(conc, resp, fitmethod, pars, sds, out, args, guess)
    
    return out, to_fit


def fitexp1():
    pass