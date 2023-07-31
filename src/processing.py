import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from fit_models import get_params
from pipeline_helper import LOG_FOLDER_PATH, track_fitted_params, get_msg_with_elapsed_time, status, custom_format, \
    print_
from acy import acy
from bmd_bounds import bmd_bounds
from tcpl_hit_helper import nest_select, hit_cont_inner
from tcpl_fit_helper import fit_curve


def processing(df, cutoff, config):
    def tcplfit_core(group):
        conc = np.array(group['concentration_unlogged'])
        resp = np.array(group['response'])
        out = {}
        for model in config['fit_models']:
            get_out_skeleton(model, out)
            try:
                fit_curve(model, conc, resp, out[model])
            except Exception as e:
                print(f"{model} >>> Error fit_curve: {e}")
        return out

    def get_out_skeleton(model, out):
        out[model] = {'pars': {p: None for p in get_params(model)},
                      **{p: None for p in ['aic', 'modl', 'rmse']}}

    def process_row(row):
        conc = np.array(row['concentration_unlogged'])
        resp = np.array(row['response'])
        rmds = np.array([np.median(resp[conc == c]) for c in np.unique(conc)])
        to_fit = (rmds.size >= 4) and np.any(np.abs(rmds) >= cutoff)
        model = 'cnst'
        out = {}
        get_out_skeleton(model, out)
        fit_curve(model, conc, resp, out[model])
        return out, to_fit

    df = preprocess(df, config)

    print_(f"{status('laptop')} Processing {df.shape[0]} concentration-response series "
           f"using {len(config['fit_models'])} different fit models:")

    time.sleep(0.05)

    desc = get_msg_with_elapsed_time(f"{status('petri_dish')}    - First run (filtering):  ", color_only_time=False)
    total = df.shape[0]
    iterator = tqdm(df.iterrows(), total=total, desc=desc, bar_format=custom_format)

    if config["n_jobs"] != 1:
        fitparams_cnst, fits = map(list, zip(*Parallel(n_jobs=config['n_jobs'])(
            delayed(process_row)(row) for _, row in iterator)))
    else:
        fitparams_cnst = []
        fits = []
        fitparams = []
        for _, row in iterator:
            result, fit = process_row(row)
            fitparams_cnst.append(result)
            fits.append(fit)

    relevant_df = df[fits]
    total = relevant_df.shape[0]
    desc = get_msg_with_elapsed_time(f"{status('atom_symbol')}    - Second run (curve-fit): ", color_only_time=False)
    iterator = tqdm(relevant_df.iterrows(), total=total, desc=desc, bar_format=custom_format)

    if config["n_jobs"] != 1:
        fitparams = Parallel(n_jobs=config['n_jobs'])(
            delayed(tcplfit_core)(row) for _, row in iterator)
    else:  # Serial version for debugging
        for _, row in iterator:
            fitparams.append(tcplfit_core(row))

    masked = np.array([{} for _ in range(len(fitparams_cnst))])
    masked[fits] = fitparams
    fitparams = [{**dict1, **dict2} for dict1, dict2 in zip(fitparams_cnst, masked)]
    df = df.assign(fitparams=fitparams)

    # Create a log file to track the parameter estimates
    with open(os.path.join(LOG_FOLDER_PATH, "params_tracked.out"), "w") as log_file:
        for res in fitparams:
            for model, params in res.items():
                if model != 'cnst':
                    params_str = ", ".join(map(str, list(params['pars'].values())))
                    log_file.write(f"{model}: {params_str}\n")

    if config['apply_track_fitted_params']:
        track_fitted_params()

    # Hit
    total = df.shape[0]
    desc = get_msg_with_elapsed_time(f"{status('test_tube')}    - Third run (hit-call):   ", color_only_time=False)
    iterator = tqdm(df.iterrows(), desc=desc, total=total, bar_format=custom_format)

    if config["n_jobs"] != 1:
        res = pd.DataFrame(Parallel(n_jobs=config['n_jobs'])(
            delayed(tcpl_hit_core)(
                params=row.fitparams,
                conc=np.array(row.concentration_unlogged),
                resp=np.array(row.response),
                cutoff=cutoff
            ) for _, row in iterator
        ))
    else:
        res = df.apply(lambda row: tcpl_hit_core(params=row.fitparams, conc=np.array(row.concentration_unlogged),
                                                 resp=np.array(row.response), cutoff=cutoff), axis=1,
                       result_type='expand')

    df[res.columns] = res
    return df


def preprocess(df, config):
    # if 'bmed' not in dat.columns:
    #     dat = dat.assign(bmed=None)
    # if 'osd' not in dat.columns:
    #     dat = dat.assign(osd=None)
    grouped = df.groupby(['aeid', 'spid'])
    df = grouped.agg(
        bmad=('bmad', np.min),
        # osd=('osd', np.min),
        # bmed=('bmed', lambda x: 0 if x.isnull().values.all() else np.max(x)),
        concentration_unlogged=('logc', lambda x: list(10 ** x)),
        response=('resp', list),
        m3ids=('m3id', list)
    ).reset_index()

    # Filter export_csv rows with NaN values in the concentration column
    df = df[df.concentration_unlogged.apply(lambda x: not any(pd.isna(x)))]

    def shrink_concentrations_and_responses(row):
        concentration_list = row['concentration_unlogged']
        response_list = row['response']
        if len(concentration_list) > config['threshold_num_datapoints']:
            unique_concentrations = pd.unique(concentration_list)
            # Calculate the median of responses over the unique concentrations
            median_responses = [pd.Series(response_list)[concentration_list == c].median() for c in
                                unique_concentrations]
            return list(unique_concentrations), list(median_responses)
        else:
            return list(concentration_list), list(response_list)

    # Shrink series with too high number of datapoints to store/handle like positive control chemical
    df['concentration_unlogged'], df['response'] = zip(*df.apply(shrink_concentrations_and_responses, axis=1))

    return df


def tcpl_hit_core(params, conc, resp, cutoff, onesd=1, bmr_scale=1.349, bmed=0, bmd_low_bnd=None, bmd_up_bnd=None):
    # Todo: ensure if fit_model == const then do not compute more than needed
    fitout = {}
    modpars = None
    modelnames = params.keys()
    aics = {m: params[m]["aic"] for m in modelnames}

    aics_values = list(aics.values())
    aics_keys = list(aics.keys())
    conc = np.array(conc)
    resp = np.array(resp)

    # use nested chisq to choose between poly1 and poly2, remove poly2 if it fails. pvalue hardcoded to .05
    if "poly1" in aics_keys and "poly2" in aics_keys:
        aics = nest_select(aics, "poly1", "poly2", dfdiff=1, pval=0.05)

    # if all fits, except the constant fail, use none for the fit method
    # when continuous hit calling is in use
    if sum(item is not None for item in aics_values) == 1 and "cnst" in aics_keys:
        fit_model = "none"
    else:
        # get AIC weights of winner (vs constant) for continuous hitcalls
        # never choose constant as winner for continuous hitcalls
        nocnstaics = {model: aics[model] for model in aics if model != "cnst"}
        fit_model = min(nocnstaics, key=nocnstaics.get)
        try:  # Todo: RuntimeWarning: invalid value encountered in scalar divide
            caikwt = np.exp(-aics["cnst"] / 2) / (np.exp(-aics["cnst"] / 2) + np.exp(-aics[fit_model] / 2))
        except:
            try:
                term = np.exp(aics["cnst"] / 2 - aics[fit_model] / 2)
                if term == np.inf:
                    caikwt = 0
                else:
                    caikwt = 1 / (1 + term)
            except Exception as e:
                print(f"Error caikwt: {e}")
                caikwt = 0

    # if the fit_model is not reported as 'none', obtain model information
    if fit_model != "none":
        fitout = params[fit_model]
        top = fitout["top"]
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

        er = fitout['pars']["er"]
        hitcall = hit_cont_inner(conc, resp, top, cutoff, er, ps=modpars, fit_model=fit_model, caikwt=caikwt, mll=mll)

    if np.isnan(hitcall):
        hitcall = 0

    ac50 = None
    bmd = None
    bmr = onesd * bmr_scale  # magic bmr is default 1.349
    if hitcall > 0:
        # fill ac's; can put after hit logic
        ac50 = acy(.5 * top, modpars, fit_model=fit_model)
        acc = acy(np.sign(top) * cutoff, modpars | {"top": top}, fit_model=fit_model)
        ac1sd = acy(np.sign(top) * onesd, modpars, fit_model=fit_model)
        bmd = acy(np.sign(top) * bmr, modpars, fit_model=fit_model)

        # get bmdl and bmdu
        try:
            bmdl = bmd_bounds(fit_model, bmr=np.sign(top) * bmr, pars=modpars, conc=conc, resp=resp, onesidedp=0.05,
                              bmd=bmd, which_bound="lower")
            bmdu = bmd_bounds(fit_model, bmr=np.sign(top) * bmr, pars=modpars, conc=conc, resp=resp, onesidedp=0.05,
                              bmd=bmd, which_bound="upper")
        except Exception as e:
            # print(f"bmd_bounds: {e}")
            pass

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
    name_list = ["fit_model", "cutoff", "bmr", "bmdl", "bmdu", "caikwt",
                 "mll", "hitcall", "ac50", "top", "acc", "ac1sd", "bmd"]

    computed_vars = list(locals().keys())
    out_list = [x for x in name_list if x in computed_vars]

    for name in out_list:
        out[name] = locals()[name]

    out = {k: v for k, v in out.items() if v is not None or 'bmd'}
    return out
