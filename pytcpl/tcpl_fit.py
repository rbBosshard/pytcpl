import numpy as np
from joblib import Parallel, delayed

from tcpl_fit_helper import curve_fit


def tcpl_fit(dat, fit_models, bidirectional=True, force_fit=False, parallelize=True, verbose=False):
    if 'bmed' not in dat.columns:
        dat = dat.assign(bmed=np.nan)
    if 'osd' not in dat.columns:
        dat = dat.assign(osd=np.nan)

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

    def tcplfit_core(group):
        conc = np.array(group['concentration_unlogged'].iloc[0])
        resp = np.array(group['response'].iloc[0])
        cutoff = group['bmad'].iloc[0]

        logc = np.log10(conc)
        unique_logc = np.unique(logc)
        rmds = np.array([np.median(resp[logc == c]) for c in unique_logc])

        if np.max(resp) == np.min(resp) and resp[0] == 0:
            print("all response values are 0: add epsilon (1e-6) to all response elements.")
            resp += 1e-6

        out = {}
        for model in fit_models:
            to_fit = len(rmds) >= 4 and (np.any(np.abs(rmds) >= cutoff) or force_fit or model == "cnst")
            out[model] = curve_fit(model, conc, resp, bidirectional, to_fit, verbose)

        return out

    fitparams = []

    if parallelize:
        fitparams = Parallel(n_jobs=-1)(delayed(tcplfit_core)(group) for _, group in dat.groupby('spid'))
    else:  # Serial version for debugging
        for _, group in dat.groupby('spid'):
            fitparams.append(tcplfit_core(group))

    dat["fitparams"] = fitparams
    return dat

