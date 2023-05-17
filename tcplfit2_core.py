import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from acy import acy

from fitcnst import fitcnst
from fithill import fithill

def tcplfit2_core(conc, resp, cutoff, force_fit=False, bidirectional=True, verbose=False, do_plot=False,
                  fitmodels=("cnst", "hill", "gnls", "poly1", "poly2", "pow", "exp2", "exp3", "exp4", "exp5")):

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
    modelnames = ["cnst"]

    for model in modelnames:
        # Only fit when four or more concentrations, the model is in fitmodels, and
        # (either one response is above cutoff OR force_fit is True OR it's the constant model.)
        to_fit = len(rmds) >= 4 and model in fitmodels and (np.any(np.abs(rmds) >= cutoff) or force_fit or model == "cnst")
        fname = "fit" + model  # requires each model function have name "fit____" where ____ is the model name

        if to_fit:
            # Use curve_fit to fit the model function
            fit_func = globals()[fname]  # Get the function object by name
            # print(fit_func)
            model_results = fit_func(conc, resp, nofit = not to_fit) #, **kwargs)
            # Create a dictionary to store the model results
            # model_results = {"conc": conc, "resp": resp, "params": popt, "covariance": pcov, "success": True}
        else:
            model_results = {"success": False}

        # Add specific calculations for each model
        if to_fit:
            if model in ("poly1", "poly2", "pow", "exp2", "exp3"):
                model_results["top"] = np.max(model_results["params"])
                model_results["ac50"] = acy(.5 * model_results["top"], model_results, type=model)
            elif model in ("hill", "exp4", "exp5"):
                model_results["top"] = model_results["params"][0]
                model_results["ac50"] = model_results["params"][1]
            elif model == "gnls":
                model_results["top"] = acy(0, model_results, type=model, returntop=True)
                if np.isnan(model_results["top"]):
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

    # # Plot all models if do_plot is True and there is at least one successful model
    # shortnames = [model for model in modelnames if model != "cnst"]
    # successes = [locals()[model]["success"] for model in shortnames]
    # if do_plot and all(successes):
    #     sorted_indices = np.argsort(logc)
    #     resp_sorted = resp[sorted_indices]
    #     logc_sorted = logc[sorted_indices]

    #     plt.figure()
    #     plt.scatter(logc_sorted, resp_sorted, color="black", label="resp")

    #     for model in shortnames:
    #         modl_sorted = locals()[model]["modl"][sorted_indices]
    #         plt.plot(logc_sorted, modl_sorted, label=model)

    #     plt.legend(loc="upper left")
    #     plt.show()

    # Put all the model outputs into one dictionary and return
    for modelname in modelnames:
        out = {model: locals()[modelname]}
    out["modelnames"] = modelnames
    # out.update(kwargs)
    return out
