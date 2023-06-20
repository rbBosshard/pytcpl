import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from fit_models import get_params
from tcpl_fit_helper import fit_curve

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def tcpl_fit(dat, fit_models, bidirectional=True, force_fit=False, parallelize=True, n_jobs=-1, test=0, verbose=False):
    def fit_curve_wrapper(model, conc, cutoff, out, resp, rmds):
        to_fit = len(rmds) >= 4 and (np.any(np.abs(rmds) >= cutoff) or force_fit or model == "cnst")
        params = get_params(model)
        out[model] = {"pars": {p: None for p in params}, "sds": {p + "_sd": None for p in params}, "modl": [],
                      **{p: None for p in ["success", "aic", "cov", "rme"]}}
        if to_fit:
            fit_curve(model, conc, resp, bidirectional, out[model], verbose)
        else:
            pass  # to_fit is sometimes false, set breakpoint/print message

    def tcplfit_core(group):
        conc = np.array(group['concentration_unlogged'])
        resp = np.array(group['response'])
        cutoff = group['bmad']

        unique_conc = np.unique(conc)
        rmds = np.array([np.median(resp[conc == c]) for c in unique_conc])

        if np.max(resp) == np.min(resp) and resp[0] == 0:
            print("all response values are 0: add epsilon (1e-6) to all response elements.")
            resp += 1e-6

        out = {}
        for model in fit_models:
            fit_curve_wrapper(model, conc, cutoff, out, resp, rmds)

        # with ThreadPoolExecutor() as executor:
        #     for model in fit_models:
        #         executor.submit(fit_curve_wrapper, model, conc, cutoff, out, resp, rmds)

        return out

    dat = preprocess(dat)
    dat = dat.head(test) if test else dat  # work only with subset if test > 1

    fitparams = []
    if parallelize:
        fitparams = Parallel(n_jobs=n_jobs)(
            delayed(tcplfit_core)(row) for _, row in tqdm(dat.iterrows(), desc='Fitting curves progress: '))
    else:  # Serial version for debugging
        for _, row in tqdm(dat.iterrows(), desc='Fitting curves progress: '):
            fitparams.append(tcplfit_core(row))

    dat = dat.assign(fitparams=fitparams)
    return dat


def preprocess(dat):
    if 'bmed' not in dat.columns:
        dat = dat.assign(bmed=None)
    if 'osd' not in dat.columns:
        dat = dat.assign(osd=None)
    grouped = dat.groupby(['aeid', 'spid', 'logc'])
    dat['rmns'] = grouped['resp'].transform(np.mean)
    dat['rmds'] = grouped['resp'].transform(np.median)
    dat['nconc'] = grouped['logc'].transform('count')
    dat['med_rmds'] = dat['rmds'] >= (3 * dat['bmad'])
    grouped = dat.groupby(['aeid', 'spid'])
    dat = grouped.agg(
        bmad=('bmad', np.min),
        resp_max=('resp', np.max),
        osd=('osd', np.min),
        bmed=('bmed', lambda x: 0 if x.isnull().values.all() else np.max(x)),
        resp_min=('resp', np.min),
        max_mean=('rmns', np.max),
        max_mean_conc=('rmns', lambda x: dat.logc[x.idxmax()]),
        max_med=('rmds', np.max),
        max_med_conc=('rmds', lambda x: dat.logc[x.idxmax()]),
        logc_max=('logc', np.max),
        logc_min=('logc', np.min),
        nconc=('logc', 'nunique'),
        npts=('resp', 'count'),
        nrep=('nconc', np.median),
        nmed_gtbl=('med_rmds', lambda x: np.sum(x) / grouped['nconc'].first().iloc[0]),
        concentration_unlogged=('logc', lambda x: list(10 ** x)),
        response=('resp', list),
        m3ids=('m3id', list)
    ).reset_index()
    grouped = dat.groupby('aeid')
    dat['tmpi'] = grouped['m3ids'].transform(lambda x: np.arange(len(x), 0, -1))
    return dat
