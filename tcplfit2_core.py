import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from acy import acy

from fitcnst import fitcnst
from fitpoly1 import fitpoly1
from fitpoly2 import fitpoly2
from fithill import fithill

def tcplfit2_core(conc, resp, cutoff, fitmodels, bidirectional, verbose= False, force_fit=False):
    conc = conc.iloc[0]
    resp = resp.iloc[0]
    cutoff = cutoff.iloc[0]

    logc = np.log10(conc)
    df = pd.DataFrame({'indexes': logc, 'values': resp})
    df.pivot_table(values='values', index='indexes', aggfunc='median')
    rmds = df.get('values')
    # rmds = np.median(resp) #resp.groupby(logc).median()
    fitmodels = ["cnst"] + list(set(fitmodels) - {"cnst"})  # cnst models must be present for conthits but not chosen

    # Check for edge case where all responses are equal
    if np.max(resp) == np.min(resp) and resp[0] == 0:
        print("all response values are 0: add epsilon (1e-6) to all response elements.")
        print("Response Range:", np.min(resp), np.max(resp))
        resp += 1e-6  # adding epsilon to resp vector

    # Fit each model based on conditions
    # modelnames = ["cnst", "hill", "gnls", "poly1", "poly2", "pow", "exp2", "exp3", "exp4", "exp5"]
    modelnames = fitmodels

    for model in modelnames:
        # Only fit when four or more concentrations, the model is in fitmodels, and
        # (either one response is above cutoff OR force_fit is True OR it's the constant model.)
        to_fit = len(rmds) >= 4 and model in fitmodels and (np.any(np.abs(rmds) >= cutoff) or force_fit or model == "cnst")
        fname = "fit" + model  # requires each model function have name "fit____" where ____ is the model name        
        # Use curve_fit to fit the model function
        fit_func = globals()[fname]  # Get the function object by name

        model_results = fit_func(np.array(conc), np.array(resp), nofit = not to_fit)

        # Add specific calculations for each model
        if to_fit:
            # print("model_results:", model_results)  
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
                    if verbose:
                        print("'top' for 'gnls' is not able to be calculated returning NA. "
                              "AC50 for gain and loss directions are returned as NA.")
                    model_results["ac50"] = np.nan
                    model_results["ac50_loss"] = np.nan
                else:
                    model_results["ac50"] = acy(.5 * model_results["top"], model_results, type=model)
                    model_results["ac50_loss"] = acy(.5 * model_results["top"], model_results, type=model, getloss=True)

        # Assign the model results to the corresponding variable name
        locals()[model] = model_results

    # Print AIC values if verbose is True
    if verbose:
        print("AIC values:")
        aics = {model: locals()[model]["aic"] for model in modelnames}
        print(aics)
        print("Winner:", min(aics, key=aics.get))

    # Put all the model outputs into one dictionary and return
    out = {}
    for modelname in modelnames:
        out[modelname] =  locals()[modelname]
    return out
