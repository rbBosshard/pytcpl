import numpy as np
from tcplfit2_core import tcplfit2_core

def tcplFit2(dat, fitmodels=["cnst", "hill", "gnls", "poly1", "poly2", "pow", "exp2", "exp3", "exp4", "exp5"], bmed=None, bidirectional=True):
    if 'bmed' not in dat.columns:
        dat = dat.assign(bmed=None)
    if 'osd' not in dat.columns:
        dat = dat.assign(osd=None)
        
    grouped = dat.groupby(['aeid', 'spid', 'logc'])
    dat['rmns'] = grouped['resp'].transform('mean')
    dat['rmds'] = grouped['resp'].transform('median')
    dat['nconcs'] = grouped['logc'].transform('count')
    dat['med_rmds'] = grouped['resp'].transform(lambda x: x.median() >= (3 * dat['bmad']))

    grouped = dat.groupby(['aeid', 'spid'])
    dat = grouped.agg(  
        bmad=('bmad', 'min'),
        resp_max=('resp', 'max'),
        osd=('osd', 'min'),
        bmed=('bmed', lambda x: 0 if x.isnull().values.all() else max(x)),
        resp_min=('resp', 'min'),
        max_mean=('rmns', 'max'),
        max_mean_conc=('rmns', lambda x: dat.logc[x.idxmax()]),
        max_med=('rmds', 'max'),
        max_med_conc=('rmds', lambda x: dat.logc[x.idxmax()]),
        logc_max=('logc', 'max'),
        logc_min=('logc', 'min'),
        nconc=('logc', 'nunique'),
        npts=('resp', 'count'),
        nrep=('nconcs', 'median'),
        nmed_gtbl = ('med_rmds', lambda x: sum(x) / grouped['nconcs'].first().iloc[0]),
        concentration_unlogged = ('logc', lambda x: list(10**(x))),
        response = ('resp', lambda x: list(x)),
        m3ids = ('m3id', lambda x: list(x))
    ).reset_index()
    
    grouped = dat.groupby('aeid')
    dat['tmpi'] = grouped['m3ids'].transform(lambda x: np.arange(len(x), 0, -1))    
    
    grouped = dat.groupby('spid')
    dat['fitparams'] = grouped[['concentration_unlogged', 'response', 'bmad']].apply(lambda x:
        tcplfit2_core(
                    conc=x['concentration_unlogged'],
                    resp=x['response'],
                    cutoff=x['bmad'],
                    bidirectional=bidirectional,
                    verbose=False,
                    force_fit=True,
                    fitmodels=fitmodels)
    ).values.tolist()
    
    return dat