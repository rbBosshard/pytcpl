import numpy as np
import pandas as pd
from math import exp

from acy import acy
from hitcontinner import hitcontinner
from hitloginner import hitloginner
from nestselect import nestselect
from bmdbounds import bmdbounds

def tcplhit2_core(params, conc, resp, cutoff, onesd, bmr_scale=1.349, bmed=0, conthits=True, aicc=False, identifiers=None, bmd_low_bnd=None, bmd_up_bnd=None):
    # a=b=tp=p=q=ga=la=er=top=ac50=ac50_loss=ac5=ac10=ac20=ac95=acc=ac1sd=bmd=None
    # bmdl=bmdu=caikwt=mll=None
    fitout = {}
    top = 0
    modelnames = params.keys()
    aics = {}
    for m in modelnames:
        aics[m] = params[m]["aic"]

    aics_values = list(aics.values())
    aics_keys = list(aics.keys())
    conc = np.array(conc)
    resp = np.array(resp)

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
        else:
            fit_method = min(aics, key=aics.get)
        # if the fit_method is not reported as 'none', obtain model information
        if fit_method != "none":
            fitout = params[fit_method]
            top = fitout["top"]
            rmse = fitout["rme"]
            modpars = fitout["pars"]
    n_gt_cutoff = np.sum(np.abs(resp) > cutoff)

    # compute discrete or continuous hitcalls
    if fit_method == "none":
        hitcall = 0
    elif conthits:
        mll = len(modpars) - aics[fit_method] / 2
        hitcall = hitcontinner(conc, resp, top, cutoff, fitout["er"],
                               ps=modpars, fit_method=fit_method,
                               caikwt=caikwt, mll=mll)
    else:
        hitcall = hitloginner(conc, resp, top, cutoff, fitout["ac50"])

    hitcall = hitcall

    if np.isnan(hitcall):
        hitcall = 0

    
    bmr = onesd * bmr_scale  # magic bmr is default 1.349
    if hitcall > 0:

        # fill ac's; can put after hit logic
        ac5 = acy(.05 * top, modpars, type=fit_method)  # note: cnst model automatically returns NAs
        ac10 = acy(.1 * top, modpars, type=fit_method)
        ac20 = acy(.2 * top, modpars, type=fit_method)
        ac95 = acy(.95 * top, modpars, type=fit_method)
        acc = acy(np.sign(top) * cutoff, modpars | {"top": top}, type=fit_method)
        ac1sd = acy(np.sign(top) * onesd, modpars, type=fit_method)
        bmd = acy(np.sign(top) * bmr, modpars, type=fit_method)

        # get bmdl and bmdu
        bmdl = bmdbounds(fit_method,
                         bmr=np.sign(top) * bmr, pars=modpars, conc=conc, resp=resp, onesidedp=0.05,
                         bmd=bmd, which_bound="lower")
        bmdu = bmdbounds(fit_method,
                         bmr=np.sign(top) * bmr, pars=modpars, conc=conc, resp=resp, onesidedp=0.05,
                         bmd=bmd, which_bound="upper")

        ################################################################33
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
    try:
        top_over_cutoff = np.abs(top) / cutoff
    except:
        top_over_cutoff = np.nan
    # conc = "|".join(conc)
    # resp = "|".join(resp)

    # out = {"cutoff": cutoff, "onesd": onesd, "bmr_scale": bmr_scale, "bmed": bmed, "conthits": conthits, "aicc": aicc, "identifiers": identifiers, "bmd_low_bnd": bmd_low_bnd, "bmd_up_bnd": bmd_up_bnd}
    out ={}
    # row contains the specified columns and any identifying, unused columns in the input
    name_list1 = [
        "n_gt_cutoff", "cutoff", "fit_method"]
    name_list2 = [
        "top_over_cutoff", "rmse", "a", "b", "tp", "p", "q", "ga", "la", "er", "bmr", "bmdl", "bmdu", "caikwt",
        "mll", "hitcall", "ac50", "ac50_loss", "top", "ac5", "ac10", "ac20", "ac95","acc", "ac1sd", "bmd"]
    name_list3 = ["conc", "resp"]

    for name in name_list1:
        out[name] = locals()[name]

    locals().update(fitout)
    if "pars" in fitout:
        locals().update(fitout["pars"])

    for name in name_list2:
        if name in locals():
            out[name] = locals()[name]
    
    out = {k: v for k, v in out.items() if v is not None}
        
    
    # row = {name: globals()[name] for name in name_list}
    # if identifiers is not None:
    #     row.update(identifiers)
    return out

