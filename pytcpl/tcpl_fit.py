import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from fit_models import get_params
from tcpl_fit_helper import fit_curve


def preprocess(dat, test):
    # if 'bmed' not in dat.columns:
    #     dat = dat.assign(bmed=None)
    # if 'osd' not in dat.columns:
    #     dat = dat.assign(osd=None)
    grouped = dat.groupby(['aeid', 'spid'])
    dat = grouped.agg(
        bmad=('bmad', np.min),
        # osd=('osd', np.min),
        # bmed=('bmed', lambda x: 0 if x.isnull().values.all() else np.max(x)),
        concentration_unlogged=('logc', lambda x: list(10 ** x)),
        response=('resp', list),
        m3ids=('m3id', list)
    ).reset_index()

    # Filter out rows with NaN values in the concentration column
    dat = dat[dat.concentration_unlogged.apply(lambda x: not any(pd.isna(x)))]
    # Shorten series with too high number of datapoints to store/handle like positive control chemical
    dat['concentration_unlogged'], dat['response'] = zip(*dat.apply(filter_concentrations_and_responses, axis=1))
    # Truncate to head according to test
    dat = dat.head(test) if test else dat  # work only with subset if test > 1

    return dat


def filter_concentrations_and_responses(row):
    concentration_list = row['concentration_unlogged']
    response_list = row['response']

    if len(concentration_list) > 1000:
        unique_concentrations = pd.unique(concentration_list)
        # Calculate the median of responses over the unique concentrations
        median_responses = [pd.Series(response_list)[concentration_list == c].median() for c in unique_concentrations]
        return list(unique_concentrations), list(median_responses)
    else:
        return list(concentration_list), list(response_list)  # Keep the original lists


def tcpl_fit(dat, fit_models, fit_strategy, cutoff, bidirectional=True, parallelize=True, n_jobs=-1, test=0):
    def tcplfit_core(group):
        conc = np.array(group['concentration_unlogged'])
        resp = np.array(group['response'])
        out = {}
        for model in fit_models:
            get_out_skeleton(model, out)
            try:
                fit_curve(model, conc, resp, bidirectional, out[model], fit_strategy)
            except Exception as e:
                print(f"{model} >>> Error fit_curve: {e}")
        return out

    def get_out_skeleton(model, out):
        out[model] = {"pars": {p: None for p in get_params(model, fit_strategy)},
                      **{p: None for p in ["aic", "modl", "rmse"]}}

    def process_row(row):
        conc = np.array(row['concentration_unlogged'])
        resp = np.array(row['response'])
        rmds = np.array([np.median(resp[conc == c]) for c in np.unique(conc)])
        to_fit = (rmds.size >= 4) and np.any(np.abs(rmds) >= cutoff)
        model = 'cnst'
        out = {}
        get_out_skeleton(model, out)
        fit_curve(model, conc, resp, bidirectional, out[model], fit_strategy)
        return out, to_fit

    dat = preprocess(dat, test)

    if os.path.exists("fit_results_log.txt"):
        os.remove("fit_results_log.txt")

    if parallelize:
        fitparams_cnst, fits = map(list, zip(*Parallel(n_jobs=n_jobs)(
            delayed(process_row)(row) for _, row in tqdm(dat.iterrows(), desc='Fit: '))))
    else:
        fitparams_cnst = []
        fits = []
        fitparams = []
        for _, row in tqdm(dat.iterrows(), desc='Fitting curves progress: '):
            result, fit = process_row(row)
            fitparams_cnst.append(result)
            fits.append(fit)

    relevant_dat = dat[fits]
    if parallelize:
        fitparams = Parallel(n_jobs=n_jobs)(
            delayed(tcplfit_core)(row) for _, row in tqdm(relevant_dat.iterrows(), desc='Fit: '))
    else:  # Serial version for debugging
        for _, row in tqdm(relevant_dat.iterrows(), desc='Fitting curves progress: '):
            fitparams.append(tcplfit_core(row))

    masked = np.array([{} for _ in range(len(fitparams_cnst))])
    masked[fits] = fitparams
    fitparams = [{**dict1, **dict2} for dict1, dict2 in zip(fitparams_cnst, masked)]
    dat = dat.assign(fitparams=fitparams)

    # Create a log file to track the parameter estimates
    with open("fit_results_log.txt", "w") as log_file:
        for res in fitparams:
            for model, params in res.items():
                if model != 'cnst':
                    params_str = ", ".join(map(str, list(params['pars'].values())))
                    log_file.write(f"{model}: {params_str}\n")

    return dat



