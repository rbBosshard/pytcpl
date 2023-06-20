import json

import numpy as np

from acy import acy
from bmd_bounds import bmd_bounds
from tcpl_hit_helper import hit_cont_inner, nest_select


def tcpl_hit_core(params, conc, resp, cutoff, onesd, bmr_scale=1.349, bmed=0, bmd_low_bnd=None, bmd_up_bnd=None):
    # Todo: ensure if fit_model == const then do not compute more than needed
    fitout = {}
    modpars = None
    modelnames = params.keys()
    aics = {m: params[m]["aic"] for m in modelnames}

    aics_values = list(aics.values())
    aics_keys = list(aics.keys())
    conc = np.array(conc)
    resp = np.array(resp)

    if sum(~np.isnan(aics_values)) == 0:
        # if all fits failed, use none for method
        fit_model = "none"
    else:
        # use nested chisq to choose between poly1 and poly2, remove poly2 if it fails.
        # pvalue hardcoded to .05
        if "poly1" in aics_keys and "poly2" in aics_keys:
            aics = nest_select(aics, "poly1", "poly2", dfdiff=1, pval=0.05)

        # if all fits, except the constant fail, use none for the fit method
        # when continuous hit calling is in use
        if sum(~np.isnan(aics_values)) == 1 and "cnst" in aics_keys:
            fit_model = "none"
        else:
            # get AIC weights of winner (vs constant) for continuous hitcalls
            # never choose constant as winner for continuous hitcalls
            nocnstaics = {model: aics[model] for model in aics if model != "cnst"}
            fit_model = min(nocnstaics, key=nocnstaics.get)
            try:
                caikwt = np.exp(-aics["cnst"] / 2) / (np.exp(-aics["cnst"] / 2) + np.exp(-aics[fit_model] / 2))
            except:
                term = np.exp(aics["cnst"] / 2 - aics[fit_model] / 2)
                if term == np.inf:
                    caikwt = 0
                else:
                    caikwt = 1 / (1 + term)

        # if the fit_model is not reported as 'none', obtain model information
        if fit_model != "none":
            fitout = params[fit_model]
            top = fitout["top"]
            rmse = fitout["rme"]
            modpars = fitout["pars"]

    n_gt_cutoff = np.sum(np.abs(resp) > cutoff)
    if cutoff != 0 and "top" in locals():
        top_over_cutoff = np.abs(top) / cutoff
    else:
        top_over_cutoff = None

    if fit_model == "none":
        hitcall = 0
    else:
        # compute continuous hitcall
        mll = len(modpars) - aics[fit_model] / 2

        hitcall = hit_cont_inner(conc, resp, top, cutoff, fitout["er"],
                                 ps=modpars, fit_model=fit_model,
                                 caikwt=caikwt, mll=mll)

    if np.isnan(hitcall):
        hitcall = 0

    ac50 = None
    ac95 = None
    bmd = None
    bmr = onesd * bmr_scale  # magic bmr is default 1.349
    if hitcall > 0:
        # fill ac's; can put after hit logic
        ac5 = acy(.05 * top, modpars, fit_model=fit_model)  # note: cnst model automatically returns NAs
        ac10 = acy(.1 * top, modpars, fit_model=fit_model)
        ac20 = acy(.2 * top, modpars, fit_model=fit_model)
        ac50 = acy(.5 * top, modpars, fit_model=fit_model)  # Todo: check: remove from here?
        ac95 = acy(.95 * top, modpars, fit_model=fit_model)
        acc = acy(np.sign(top) * cutoff, modpars | {"top": top}, fit_model=fit_model)
        ac1sd = acy(np.sign(top) * onesd, modpars, fit_model=fit_model)
        bmd = acy(np.sign(top) * bmr, modpars, fit_model=fit_model)

        # get bmdl and bmdu
        bmdl = bmd_bounds(fit_model,
                          bmr=np.sign(top) * bmr, pars=modpars, conc=conc, resp=resp, onesidedp=0.05,
                          bmd=bmd, which_bound="lower")
        bmdu = bmd_bounds(fit_model,
                          bmr=np.sign(top) * bmr, pars=modpars, conc=conc, resp=resp, onesidedp=0.05,
                          bmd=bmd, which_bound="upper")

        # apply bmd min
        if bmd_low_bnd is not None and not np.isnan(bmd):
            min_conc = np.min(conc)
            min_bmd = min_conc * bmd_low_bnd
            if bmd < min_bmd:
                bmd_diff = min_bmd - bmd
                # shift all bmd to the right
                bmd += bmd_diff
                bmdl += bmd_diff
                bmdu += bmd_diff

        # apply bmd max
        if bmd_up_bnd is not None and not np.isnan(bmd):
            max_conc = np.max(conc)
            max_bmd = max_conc * bmd_up_bnd
            if bmd > max_bmd:
                # shift all bmd to the left
                bmd_diff = bmd - max_bmd
                bmd -= bmd_diff
                bmdl -= bmd_diff
                bmdu -= bmd_diff

    locals().update(fitout)
    if "pars" in fitout:
        locals().update(fitout["pars"])

    out = {}
    name_list = ["n_gt_cutoff", "cutoff", "fit_model",
                 "top_over_cutoff", "rmse", "a", "b", "tp", "p", "q", "ga", "la", "er", "bmr", "bmdl", "bmdu", "caikwt",
                 "mll", "hitcall", "ac50", "ac50_loss", "top", "ac5", "ac10", "ac20", "ac95", "acc", "ac1sd", "bmd"]

    computed_vars = list(locals().keys())
    out_list = [x for x in name_list if x in computed_vars]

    for name in out_list:
        out[name] = locals()[name]

    out = {k: v for k, v in out.items() if v is not None}
    return out
