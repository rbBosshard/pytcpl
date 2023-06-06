import numpy as np
from joblib import Parallel, delayed

from fit_method_helper import curve_fit


def tcpl_fit2(dat, fitmodels, bidirectional=True, force_fit=False, parallelize=True, verbose=False):
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

    def tcplfit2_core(group):
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
        for model in fitmodels:
            to_fit = get_to_fit_condition(cutoff, force_fit, rmds, model)
            out[model] = curve_fit(model, conc, resp, bidirectional, to_fit, verbose)

        return out

    fitparams = []

    if parallelize:
        fitparams = Parallel(n_jobs=-1)(delayed(tcplfit2_core)(group) for _, group in dat.groupby('spid'))
    else:  # Serial version for debugging
        for _, group in dat.groupby('spid'):
            fitparams.append(tcplfit2_core(group))

    dat["fitparams"] = fitparams
    return dat


def get_to_fit_condition(cutoff, force_fit, rmds, model):
    to_fit = len(rmds) >= 4 and (np.any(np.abs(rmds) >= cutoff) or force_fit or model == "cnst")
    return to_fit


def cnst(ps, x):
    # ignores ps
    return np.zeros(len(x))


def exp2(ps, x):
    # a = ps[0], b = ps[1]
    return ps[0] * (np.exp(x / ps[1]) - 1)


def exp3(ps, x):
    # a = ps[0], b = ps[1], p = ps[2]
    return ps[0] * (np.exp((x / ps[1]) ** ps[2]) - 1)


def exp4(ps, x):
    # tp = ps[0], ga = ps[1]
    return ps[0] * (1 - 2 ** (-x / ps[1]))


def exp5(ps, x):
    # tp = ps[0], ga = ps[1], p = ps[2]
    return ps[0] * (1 - 2 ** (-(x / ps[1]) ** ps[2]))


def gnls(ps, x):
    # gnls function with regular units
    # tp = ps[0], ga = ps[1], p = ps[2], la = ps[3], q = ps[4]
    gn = 1 / (1 + (ps[1] / x) ** ps[2])
    ls = 1 / (1 + (x / ps[3]) ** ps[4])
    return ps[0] * gn * ls


def loggnls(ps, x):
    # gnls function with log units: x = log10(conc) and ga/la = log10(gain/loss ac50)
    # tp = ps[0], ga = ps[1], p = ps[2], la = ps[3], q = ps[4]
    gn = 1 / (1 + 10 ** ((ps[1] - x) * ps[2]))
    ls = 1 / (1 + 10 ** ((x - ps[3]) * ps[4]))
    return ps[0] * gn * ls


def hillfn(ps, x):
    # hill function with regular units
    # tp = ps[0], ga = ps[1], p = ps[2]
    return ps[0] / (1 + (ps[1] / x) ** ps[2])


def loghill(ps, x):
    # hill function with log units: x = log10(conc) and ga = log10(ac50)
    # tp = ps[0], ga = ps[1], p = ps[2]
    return ps[0] / (1 + 10 ** (ps[2] * (ps[1] - x)))


def poly1(ps, x):
    # a = ps[0]
    return ps[0] * x


def poly2(ps, x):
    # a = ps[0], b = ps[1]
    x0 = x / ps[1]
    return ps[0] * (x0 + x0 * x0)


def pow(ps, x):
    # a = ps[0], p = ps[1]
    return ps[0] * x ** ps[1]
