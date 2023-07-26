import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from fit_models import get_params
from pytcpl.pipeline_helper import get_cutoff
from tcpl_fit_helper import fit_curve
from tcpl_fit_helper import generate_output


def tcpl_fit(dat, fit_models, fit_strategy, key_positive_control, bidirectional=True, parallelize=True, n_jobs=-1,
             test=0):
    def tcplfit_core(group):
        conc = np.array(group['concentration_unlogged'])
        resp = np.array(group['response'])

        if np.max(resp) == np.min(resp) and resp[0] == 0:
            print("all response values are 0: add epsilon (1e-6) to all response elements.")
            resp += 1e-6

        out = {}
        for model in fit_models:
            params = get_params(model, fit_strategy)
            out[model] = {"pars": {p: None for p in params}, "sds": {p + "_sd": None for p in params}, "modl": [],
                          **{p: None for p in ["aic"]}}

            fit_curve(model, conc, resp, bidirectional, out[model], fit_strategy)

        return out

    if os.path.exists("fit_results_log.txt"):
        os.remove("fit_results_log.txt")

    dat = preprocess(dat)
    # Filter out rows with NaN values in the concentration column
    dat = dat[dat.concentration_unlogged.apply(lambda x: not any(pd.isna(x)))]
    # Truncate key_positive_control (often has too huge number of datapoints to store/handle).
    # Custom function to filter concentrations and corresponding responses
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

    dat['concentration_unlogged'], dat['response'] = zip(*dat.apply(filter_concentrations_and_responses, axis=1))
    dat = dat.head(test) if test else dat  # work only with subset if test > 1

    # Filter
    def process_row(row, cutoff, fit_strategy):
        conc = np.array(row['concentration_unlogged'])
        resp = np.array(row['response'])
        rmds = np.array([np.median(resp[conc == c]) for c in np.unique(conc)])

        out = {}
        to_fit = (rmds.size >= 4) and np.any(np.abs(rmds) >= cutoff)

        model = 'cnst'
        params = get_params(model, fit_strategy)
        out[model] = {"pars": {p: None for p in params}, "sds": {p + "_sd": None for p in params}, "modl": [],
                      **{p: None for p in ["aic"]}}

        fit = [0] if fit_strategy == "mle" else []
        generate_output(model, conc, resp, out[model], fit, fit_strategy)
        return out, to_fit

    if parallelize:
        fitparams_cnst, fits = map(list, zip(*Parallel(n_jobs=n_jobs)(
            delayed(process_row)(row, get_cutoff(aeid=row['aeid'], bmad=row['bmad']),
                                 fit_strategy) for _, row in
            tqdm(dat.iterrows(), desc='Fitting curves progress: ')
        )))

        fitparams = Parallel(n_jobs=n_jobs)(
            delayed(tcplfit_core)(row) for _, row in tqdm(dat[fits].iterrows(), desc='Fitting curves progress: '))
    else:  # Serial version for debugging
        # Create empty lists to store the results
        fitparams_cnst = []
        fits = []
        fitparams = []

        # Sequential processing of the DataFrame 'dat'
        for _, row in tqdm(dat.iterrows(), desc='Fitting curves progress: '):
            # Your existing code for processing each row goes here...
            result, fit = process_row(row, get_cutoff(aeid=row['aeid'], bmad=row['bmad']),
                                      fit_strategy)

            fitparams_cnst.append(result)
            fits.append(fit)
        for _, row in tqdm(dat[fits].iterrows(), desc='Fitting curves progress: '):
            fitparams.append(tcplfit_core(row))

    # create output array
    masked = np.array([{} for _ in range(len(fitparams_cnst))])
    masked[fits] = fitparams
    fitparams = [
        {**dict1, **dict2}
        for dict1, dict2 in zip(fitparams_cnst, masked)
    ]

    # Create a log file to track the parameter estimates
    with open("fit_results_log.txt", "w") as log_file:
        for res in fitparams:
            for model, params in res.items():
                if model != 'cnst':
                    params_str = ", ".join(map(str, list(params['pars'].values())))
                    log_file.write(f"{model}: {params_str}\n")

    dat = dat.assign(fitparams=fitparams)
    return dat


def preprocess(dat):
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
    return dat
