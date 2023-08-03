import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from acy import acy
from constants import custom_format, custom_format_
from fit_models import get_params
from pipeline_helper import track_fitted_params, get_msg_with_elapsed_time, status, print_
from tcpl_fit_helper import fit_curve
from tcpl_hit_helper import nest_select, hit_cont_inner


def processing(df, cutoff, config):
    print_(f"{status('laptop')} Processing {df.shape[0]} concentration-response series")
    df = preprocess(df, config)

    def get_out_keys(model, out):
        out[model] = {'pars': {p: None for p in get_params(model)}, **{p: None for p in ['aic', 'modl', 'rmse']}}

    def fit(group):
        out = {}
        for model in config['curve_fit_models']:
            get_out_keys(model, out)
            fit_curve(model, np.array(group['conc']), np.array(group['response']), out[model])
        return out

    def process_row(row):
        conc = np.array(row['conc'])
        resp = np.array(row['response'])
        rmds = np.array([np.median(resp[conc == c]) for c in np.unique(conc)])
        to_fit = (rmds.size >= 4) and np.any(np.abs(rmds) >= cutoff)
        model = 'cnst'
        out = {}
        get_out_keys(model, out)
        fit_curve(model, conc, resp, out[model])
        return out, to_fit

    # Preprocessing & Filtering
    desc = get_msg_with_elapsed_time(f"{status('hammer_and_wrench')}  -> Preprocessing:  ", color_only_time=False)
    it = tqdm(df.iterrows(), total=df.shape[0], desc=desc, bar_format=custom_format)
    if config["n_jobs"] != 1:
        fit_params_cnst, fits = map(list, zip(*Parallel(n_jobs=config['n_jobs'])(delayed(process_row)(i) for _, i in it)))
    else:
        fit_params_cnst, fits = map(list, zip(*(process_row(row) for _, row in it)))

    # Curve-Fitting
    desc = get_msg_with_elapsed_time(f"{status('comet')}  -> Curve-Fitting: ", color_only_time=False)
    it = tqdm(df[fits].iterrows(), total=df[fits].shape[0], desc=desc, bar_format=custom_format_)
    if config["n_jobs"] != 1:
        fit_params = Parallel(n_jobs=config['n_jobs'])(delayed(fit)(i) for _, i in it)
    else:
        fit_params = [fit(i) for _, i in it]

    # Merge fit_params
    masked = np.array([{} for _ in range(len(fit_params_cnst))])
    masked[fits] = fit_params
    fit_params = [{**dict1, **dict2} for dict1, dict2 in zip(fit_params_cnst, masked)]
    df = df.assign(fit_params=fit_params)

    if config['enable_curve_fit_parameter_tracking']:
        track_fitted_params(df['fit_params'])

    # Hit-Calling
    desc = get_msg_with_elapsed_time(f"{status('horizontal_traffic_light')}  -> Hit-Calling:   ", color_only_time=False)
    it = tqdm(df.iterrows(), desc=desc, total=df.shape[0], bar_format=custom_format)

    if config["n_jobs"] != 1:
        res = pd.DataFrame(Parallel(n_jobs=config['n_jobs'])(
            delayed(tcpl_hit_core)(
                params=i.fit_params,
                conc=np.array(i.conc),
                resp=np.array(i.response),
                cutoff=cutoff
            ) for _, i in it
        ))
    else:
        res = df.apply(lambda i: tcpl_hit_core(params=i.fit_params, conc=np.array(i.conc), resp=np.array(i.response),
                                               cutoff=cutoff), axis=1, result_type='expand')
    df[res.columns] = res
    return df


def preprocess(df, config):
    df = df.groupby(['aeid', 'spid']).agg(conc=('logc', lambda x: list(10 ** x)), response=('resp', list)).reset_index()
    df = df[df.conc.apply(lambda x: not any(pd.isna(x)))]
    df = df[:config['enable_data_subsetting']] if config['enable_data_subsetting'] else df

    def shrink_series(row):
        conc = row['conc']
        resp = row['response']
        uconc = pd.unique(conc)
        t = len(conc) > config['num_datapoints_per_series_threshold']
        return (list(uconc), [pd.Series(resp)[conc == c].median() for c in uconc]) if t else (list(conc), list(resp))

    # Shrink series containing too many datapoints to handle. Often the case for positive control chemical
    df['conc'], df['response'] = zip(*df.apply(shrink_series, axis=1))

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
            assert aics[fit_model] <= aics["cnst"]
            caikwt = np.exp((aics[fit_model] - aics["cnst"]) / 2)
        except Exception as e:
            print(f"Error caikwt: {e}")
            caikwt = 1

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
        # try:
        #     bmdl = bmd_bounds(fit_model, bmr=np.sign(top) * bmr, pars=modpars, conc=conc, resp=resp, onesidedp=0.05,
        #                       bmd=bmd, which_bound="lower")
        #     bmdu = bmd_bounds(fit_model, bmr=np.sign(top) * bmr, pars=modpars, conc=conc, resp=resp, onesidedp=0.05,
        #                       bmd=bmd, which_bound="upper")
        # except Exception as e:
        #     print(f"bmd_bounds: {e}")
        #     pass

        # apply bmd min
        # if bmd_low_bnd is not None and not np.isnan(bmd):
        #     min_conc = np.min(conc)
        #     min_bmd = min_conc * bmd_low_bnd
        #     if bmd < min_bmd:
        #         bmd_diff = min_bmd - bmd
        #         # shift all bmd to the right
        #         bmd += bmd_diff
        #         bmdl += bmd_diff
        #         bmdu += bmd_diff
        #
        # # apply bmd max
        # if bmd_up_bnd is not None and not np.isnan(bmd):
        #     max_conc = np.max(conc)
        #     max_bmd = max_conc * bmd_up_bnd
        #     if bmd > max_bmd:
        #         # shift all bmd to the left
        #         bmd_diff = bmd - max_bmd
        #         bmd -= bmd_diff
        #         bmdl -= bmd_diff
        #         bmdu -= bmd_diff

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
