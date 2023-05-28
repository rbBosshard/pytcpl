import pandas as pd
import numpy as np

from acy import acy
from tcplHit2_core import tcplhit2_core
from tcplLoadData import tcplLoadData

def tcplHit2(mc4, coff):
    df = mc4[mc4['model'] != 'all']
    nested_mc4 = df.groupby('m4id').apply(lambda x: tcplFit2_nest(pd.DataFrame({'model': x['model'], 'model_param' : x['model_param'], 'model_val': x['model_val']}))).reset_index(name = 'params')
    val = list(nested_mc4['m4id'].values)
    l4_agg = tcplLoadData(lvl='agg', fld='m4id', val=val)
    val = list(l4_agg['m3id'].values) # Might be a huge list: Maybe use BETWEEN sql operator to get all values between two values
    data = tcplLoadData(lvl=3, fld='m3id', val=val)
    l3_dat = pd.merge(l4_agg, data, on=['aeid', 'm3id', 'm2id', 'm1id', 'm0id', 'spid', 'logc', 'resp'], how='left')

    nested_mc4 = nested_mc4.merge(l3_dat.groupby("m4id").agg(conc=("logc", lambda x: list(10**x)), resp=("resp", lambda x: list(x))).reset_index(), on="m4id", how="left")

    nested_mc4 = nested_mc4.merge(mc4.query('model_param == "onesd"')[["m4id", "model_val"]].rename(columns={"model_val": "onesd"}), on="m4id", how="inner")

    nested_mc4 = nested_mc4.merge(mc4.query('model_param == "bmed"')[["m4id", "model_val"]].rename(columns={"model_val": "bmed"}), on="m4id", how="inner")


    test = (
        nested_mc4.apply(lambda row: pd.Series(tcplhit2_core(row['params'], 
                                                             row['conc'], 
                                                             row['resp'], 
                                                             row['bmed'], coff, row['onesd'])), axis=1)
        .join(nested_mc4.drop(['conc', 'resp'], axis=1))
    )

    res = pd.concat([test, pd.DataFrame(test['df'].tolist())], axis=1)


    res['ac95'] = res.apply(lambda row: acy(0.95 * row['top'], 
                                            {'a': row['a'], 'b': row['b'], 
                                             'ga': row['ga'], 'la': row['la'], 
                                             'p': row['p'], 'q': row['q'], 
                                             'tp': row['tp']}), axis=1)

    res['coff_upper'] = 1.2 * res['cutoff']
    res['coff_lower'] = 0.8 * res['cutoff']

    mc4_subset = mc4[['m4id', 'logc_min', 'logc_max']].drop_duplicates()

    res = pd.merge(res, mc4_subset, on='m4id', how='left')

    res['fitc'] = np.select(
        [
            (res['hitcall'] >= 0.9) & (np.abs(res['top']) <= res['coff_upper']) & (res['ac50'] <= res['logc_min']),
            (res['hitcall'] >= 0.9) & (np.abs(res['top']) <= res['coff_upper']) & (res['ac50'] > res['logc_min']) & (res['ac95'] < res['logc_max']),
            (res['hitcall'] >= 0.9) & (np.abs(res['top']) <= res['coff_upper']) & (res['ac50'] > res['logc_min']) & (res['ac95'] >= res['logc_max']),
            (res['hitcall'] >= 0.9) & (np.abs(res['top']) > res['coff_upper']) & (res['ac50'] <= res['logc_min']),
            (res['hitcall'] >= 0.9) & (np.abs(res['top']) > res['coff_upper']) & (res['ac50'] > res['logc_min']) & (res['ac95'] < res['logc_max']),
            (res['hitcall'] >= 0.9) & (np.abs(res['top']) > res['coff_upper']) & (res['ac50'] > res['logc_min']) & (res['ac95'] >= res['logc_max']),
            (res['hitcall'] < 0.9) & (np.abs(res['top']) < res['coff_lower']),
            (res['hitcall'] < 0.9) & (np.abs(res['top']) >= res['coff_lower']),
            (res['fit_method'] == 'none')
        ],
        [36, 37, 38, 40, 41, 42, 13, 15, 2],
        default=np.nan
    )

    mc5 = pd.merge(
        res[['m4id', 'aeid', 'fit_method', 'hitcall', 'fitc', 'cutoff']].rename(columns={'fit_method': 'modl', 'hitcall': 'hitc', 'fitc': 'fitc', 'cutoff': 'coff'}),
        mc4[['m4id', 'aeid']].drop_duplicates(),
        on=['m4id', 'aeid'],
        how='left'
    )
    mc5['model_type'] = 2

    mc5_param = pd.merge(
        res[['m4id', 'aeid', 'top_over_cutoff', 'bmd']].rename(columns={'top_over_cutoff': 'hit_param', 'bmd': 'hit_val'}),
        mc4[['m4id', 'aeid']].drop_duplicates(),
        on=['m4id', 'aeid'],
        how='left'
    )
    mc5_param = pd.melt(mc5_param, id_vars=['m4id', 'aeid'], value_vars=['hit_param', 'hit_val'], 
                        var_name='hit_param', value_name='hit_val')
    mc5_param = mc5_param.dropna(subset=['hit_val'])

    mc5 = pd.merge(mc5, mc5_param, on=['m4id', 'aeid'], how='inner')

    return mc5

def tcplFit2_nest(dat):
    modelnames = dat["model"].unique()

    dicts = {}

    for m in modelnames:
        dicts[m] = dict(tuple(dat[dat["model"] == m].groupby("model_param")["model_val"]))
        
    for m in modelnames:
        if m == "cnst":
            modpars = ["er"]
        elif m == "exp2":
            modpars = ["a", "b", "er"]
        elif m == "exp3":
            modpars = ["a", "b", "p", "er"]
        elif m == "exp4":
            modpars = ["tp", "ga", "er"]
        elif m == "exp5":
            modpars = ["tp", "ga", "p", "er"]
        elif m == "hill":
            modpars = ["tp", "ga", "p", "er"]
        elif m == "poly1":
            modpars = ["a", "er"]
        elif m == "poly2":
            modpars = ["a", "b", "er"]
        elif m == "pow":
            modpars = ["a", "p", "er"]
        elif m == "gnls":
            modpars = ["tp", "ga", "p", "la", "q", "er"]

        dicts[m]["pars"] = modpars

    
    print("?????")
    print([dicts[m] for m in modelnames])
    
    out = dicts #+ [{"modelnames": modelnames}] # out = list with list[0] = values and list[1]=keys

    return out