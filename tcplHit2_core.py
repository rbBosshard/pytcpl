import numpy as np
import pandas as pd
from math import exp

from acy import acy
from hitcontinner import hitcontinner
from hitloginner import hitloginner
from nestselect import nestselect
from bmdbounds import bmdbounds

def tcplhit2_core(params, conc, resp, cutoff, onesd, bmr_scale=1.349, bmed=0, conthits=True, aicc=False, identifiers=None, bmd_low_bnd=None, bmd_up_bnd=None):
    # initialize parameters to None
    a = b = tp = p = q = ga = la = er = top = ac50 = ac50_loss = ac5 = ac10 = ac20 = acc = ac1sd = bmd = None
    bmdl = bmdu = caikwt = mll = None
    # get AICs and degrees of freedom
    # pd.set_option("display.max_rows", n)
    # pd.set_option("display.expand_frame_repr", True)
    # pd.set_option('display.width', 1000)

 
    modelnames = params.keys()
    
    # aics = [list(params[model]["aic"].values) for model in modelnames if "aic" in params[model]][0]
    aics = {}
    dfs = {}
    for m in modelnames:
        aics[m] = list(params[m]["aic"].values)

    aics_values = list(aics.values())[0]
    aics_keys = list(aics.keys())

    if sum(~np.isnan(aics_values)) == 0:
        # if all fits failed, use none for method
        fit_method = "none"
    else:
        # use nested chisq to choose between poly1 and poly2, remove poly2 if it fails.
        # pvalue hardcoded to .05
        if "poly1" in aics_keys and "poly2" in aics_keys:
            aics = nestselect(aics, "poly1", "poly2", dfdiff=1, pval=0.05)
      
        if conthits:
            # if all fits, except the constant fail, use none for the fit method
            # when continuous hit calling is in use
            print(f"aics_keys: {aics_keys}")
            if sum(~np.isnan(aics_values)) == 1 and "cnst" in aics_keys:
                fit_method = "none"
                rmse = None
            else:
                # get AIC weights of winner (vs constant) for continuous hitcalls
                # never choose constant as winner for continuous hitcalls
                nocnstaics = {model: aics[model] for model in aics if model != "cnst"}
                fit_method = min(nocnstaics, key=nocnstaics.get)
                caikwt = exp(-aics["cnst"] / 2) / (exp(-aics["cnst"] / 2) + exp(-aics[fit_method] / 2))
                if np.isnan(caikwt):
                    term = exp(aics["cnst"] / 2 - aics[fit_method] / 2)
                    if term == np.inf:
                        caikwt = 0
                    else:
                        caikwt = 1 / (1 + term)
                        # caikwt = 1
        else:
            fit_method = min(aics, key=aics.get)
        # if the fit_method is not reported as 'none', obtain model information
        if fit_method != "none":
            fitout = params[fit_method]
            rmse = fitout["rme"]
            modpars = fitout["pars"]
            for key, value in fitout.items():
                globals()[key] = value
    n_gt_cutoff = np.sum(np.abs(resp) > cutoff)

    # compute discrete or continuous hitcalls
    if fit_method == "none":
        hitcall = 0
    elif conthits:
        mll = len(modpars) - aics[fit_method] / 2
        hitcall = hitcontinner(conc, resp, top, cutoff, er,
                               ps=modpars, fit_method=fit_method,
                               caikwt=caikwt, mll=mll)
    else:
        hitcall = hitloginner(conc, resp, top, cutoff, ac50)

    print(f"hitcall: {hitcall}")
    print(f"fit_method: {fit_method}")

    if np.isnan(hitcall):
        hitcall = 0

    bmr = onesd * bmr_scale  # magic bmr is default 1.349
    if hitcall > 0:

        # fill ac's; can put after hit logic
        ac5 = acy(.05 * top, modpars, type=fit_method)  # note: cnst model automatically returns NAs
        ac10 = acy(.1 * top, modpars, type=fit_method)
        ac20 = acy(.2 * top, modpars, type=fit_method)
        acc = acy(np.sign(top) * cutoff, modpars + [top], type=fit_method)
        ac1sd = acy(np.sign(top) * onesd, modpars, type=fit_method)
        bmd = acy(np.sign(top) * bmr, modpars, type=fit_method)

        # get bmdl and bmdu
        bmdl = bmdbounds(fit_method,
                         bmr=np.sign(top) * bmr, pars=modpars, conc=conc, resp=resp, onesidedp=0.05,
                         bmd=bmd, which_bound="lower")
        bmdu = bmdbounds(fit_method,
                         bmr=np.sign(top) * bmr, pars=modpars, conc=conc, resp=resp, onesidedp=0.05,
                         bmd=bmd, which_bound="upper")

        # apply bmd min
        if bmd_low_bnd is not None and not np.isnan(bmd):
            min_conc = min(conc)
            min_bmd = min_conc * bmd_low_bnd
            if bmd < min_bmd:
                bmd_diff = min_bmd - bmd
                # shift all bmd to the right
                bmd += bmd_diff
                bmdl += bmd_diff
                bmdu += bmd_diff

        # apply bmd max
        if bmd_up_bnd is not None and not np.isnan(bmd):
            max_conc = max(conc)
            max_bmd = max_conc * bmd_up_bnd
            if bmd > max_bmd:
                # shift all bmd to the left
                bmd_diff = bmd - max_bmd
                bmd -= bmd_diff
                bmdl -= bmd_diff
                bmdu -= bmd_diff

    top_over_cutoff = np.abs(top) / cutoff
    conc = "|".join(conc)
    resp = "|".join(resp)

    # row contains the specified columns and any identifying, unused columns in the input
    name_list = [
        "n_gt_cutoff", "cutoff", "fit_method",
        "top_over_cutoff", "rmse", "a", "b", "tp", "p", "q", "ga", "la", "er", "bmr", "bmdl", "bmdu", "caikwt",
        "mll", "hitcall", "ac50", "ac50_loss", "top", "ac5", "ac10", "ac20", "acc", "ac1sd", "bmd", "conc", "resp"]
    
    row = {name: globals()[name] for name in name_list}
    if identifiers is not None:
        row.update(identifiers)

    return row

