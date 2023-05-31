import numpy as np
import pandas as pd
from tcplfit2_core import tcplfit2_core
from multiprocessing import Pool

def tcplFit2(dat, fitmodels, bidirectional=True):
    if 'bmed' not in dat.columns:
        dat = dat.assign(bmed=None)
    if 'osd' not in dat.columns:
        dat = dat.assign(osd=None)
    
        
    grouped = dat.groupby(['aeid', 'spid', 'logc'])
    dat['rmns'] = grouped['resp'].transform(np.mean)
    dat['rmds'] = grouped['resp'].transform(np.median)
    dat['nconcs'] = grouped['logc'].transform('count')
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
        nrep=('nconcs', np.median),
        nmed_gtbl = ('med_rmds', lambda x: np.sum(x) / grouped['nconcs'].first().iloc[0]),
        concentration_unlogged = ('logc', lambda x: list(10**(x))),
        response = ('resp', list),
        m3ids = ('m3id', list)
    ).reset_index()
    
    grouped = dat.groupby('aeid')
    dat['tmpi'] = grouped['m3ids'].transform(lambda x: np.arange(len(x), 0, -1))    


    # grouped = dat.groupby('spid')[['concentration_unlogged', 'response', 'bmad']]

    # dat['fitparams'] = grouped.apply(lambda x:
    #     tcplfit2_core(
    #                 conc=x['concentration_unlogged'],
    #                 resp=x['response'],
    #                 cutoff=x['bmad'],
    #                 fitmodels=fitmodels,
    #                 bidirectional=bidirectional,
    #                 verbose=False,
    #                 force_fit=True,
    #                 )
    # ).values.tolist()

    
    # return dat


    # def process_group(x):
    #     return tcplfit2_core(
    #         conc=x['concentration_unlogged'],
    #         resp=x['response'],
    #         cutoff=x['bmad'],
    #         fitmodels=fitmodels,
    #         bidirectional=bidirectional,
    #         verbose=False,
    #         force_fit=True,
    #     )


    # dat['fitparams'] = grouped.swifter.apply(lambda x: process_group(x), axis=1)
    from joblib import Parallel, delayed

    def compute_fitparams(group):
        out = tcplfit2_core(np.array(group['concentration_unlogged'].iloc[0]),
                            np.array(group['response'].iloc[0]),
                            cutoff=group['bmad'].iloc[0],
                            bidirectional=bidirectional,
                            verbose=False,
                            force_fit=True,
                            fitmodels=fitmodels)
        return out

    dat['fitparams'] = (
        Parallel(n_jobs=-1)(delayed(compute_fitparams)(group) for _, group in dat.groupby('spid'))
    )

    # fitparams = []
    # for _, group in dat.groupby('spid'):
    #     result = compute_fitparams(group)
    #     fitparams.append(result)
    # dat['fitparams'] = fitparams


    return dat