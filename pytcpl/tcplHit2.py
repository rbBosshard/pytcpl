import pandas as pd
import numpy as np

from tcplHit2_core import tcpl_hit2_core
from tcplLoadData import tcpl_load_data


def tcpl_hit2(mc4, coff, verbose=False):
    df = mc4[mc4['model'] != 'all']
    nested_mc4 = df.groupby('m4id').apply(lambda x: tcpl_fit2_nest(
        pd.DataFrame({'model': x['model'], 'model_param': x['model_param'], 'model_val': x['model_val']}))).reset_index(
        name='params')
    val = list(nested_mc4['m4id'].values)
    l4_agg = tcpl_load_data(lvl='agg', fld='m4id', val=val)
    # Might be a huge list: Maybe use BETWEEN sql operator to get all values between two values
    val = list(l4_agg['m3id'].values)
    data = tcpl_load_data(lvl=3, fld='m3id', val=val)
    l3_dat = pd.merge(l4_agg, data, on=['aeid', 'm3id', 'm2id', 'm1id', 'm0id', 'spid', 'logc', 'resp'], how='left')

    nested_mc4 = nested_mc4.merge(l3_dat.groupby("m4id").agg(conc=("logc", lambda x: list(10 ** x)),
                                                             resp=("resp", lambda x: list(x))).reset_index(), on="m4id",
                                  how="left")
    nested_mc4 = nested_mc4.merge(
        mc4.query('model_param == "onesd"')[["m4id", "model_val"]].rename(columns={"model_val": "onesd"}), on="m4id",
        how="inner")
    nested_mc4 = nested_mc4.merge(
        mc4.query('model_param == "bmed"')[["m4id", "model_val"]].rename(columns={"model_val": "bmed"}), on="m4id",
        how="inner")

    test = (
        nested_mc4.assign(df=lambda x: [
            tcpl_hit2_core(params=x.params, conc=np.array(x.conc), resp=np.array(x.resp), bmed=x.bmed, cutoff=coff,
                           onesd=x.onesd) for _, x in x.iterrows()]).drop(['conc', 'resp'], axis=1)
    )

    res = pd.concat([nested_mc4, pd.DataFrame(test['df'].tolist())], axis=1)

    res['coff_upper'] = 1.2 * coff
    res['coff_lower'] = 0.8 * coff

    mc4_subset = mc4[['m4id', 'logc_min', 'logc_max']].drop_duplicates()

    res = pd.merge(res, mc4_subset, on='m4id', how='left')

    res['fitc'] = np.select(
        [
            (res['hitcall'] >= 0.9) & (np.abs(res['top']) <= res['coff_upper']) & (res['ac50'] <= res['logc_min']),
            (res['hitcall'] >= 0.9) & (np.abs(res['top']) <= res['coff_upper']) & (res['ac50'] > res['logc_min']) & (
                        res['ac95'] < res['logc_max']),
            (res['hitcall'] >= 0.9) & (np.abs(res['top']) <= res['coff_upper']) & (res['ac50'] > res['logc_min']) & (
                        res['ac95'] >= res['logc_max']),
            (res['hitcall'] >= 0.9) & (np.abs(res['top']) > res['coff_upper']) & (res['ac50'] <= res['logc_min']),
            (res['hitcall'] >= 0.9) & (np.abs(res['top']) > res['coff_upper']) & (res['ac50'] > res['logc_min']) & (
                        res['ac95'] < res['logc_max']),
            (res['hitcall'] >= 0.9) & (np.abs(res['top']) > res['coff_upper']) & (res['ac50'] > res['logc_min']) & (
                        res['ac95'] >= res['logc_max']),
            (res['hitcall'] < 0.9) & (np.abs(res['top']) < res['coff_lower']),
            (res['hitcall'] < 0.9) & (np.abs(res['top']) >= res['coff_lower']),
            (res['fit_method'] == 'none')
        ],
        [36, 37, 38, 40, 41, 42, 13, 15, 2],
        default=np.nan
    )

    mc5 = pd.merge(
        res,
        mc4[['m4id', 'aeid']].drop_duplicates(),
        on=['m4id'],
        how='left'
    )
    mc5 = mc5[['m4id', 'aeid', 'fit_method', 'hitcall', 'fitc', 'cutoff']].rename(
        columns={"fit_method": "modl", "hitcall": "hitc", "cutoff": "coff"}).assign(model_type=2)

    mc5_param = pd.merge(res, mc4[['m4id', 'aeid']].drop_duplicates(), on='m4id', how='left')

    pivots = list(mc5_param.loc[:, 'top_over_cutoff':'bmd'].columns)

    mc5_param = mc5_param.melt(
        id_vars=['m4id', 'aeid'],  # Specify other columns to keep unchanged
        value_vars=pivots,  # Specify the columns to pivot
        var_name="hit_param",  # Name for the new column containing column names
        value_name="hit_val"  # Name for the new column containing column values
    )
    mc5_param = mc5_param.dropna(subset=['hit_val'])

    mc5 = pd.merge(mc5, mc5_param, on=['m4id', 'aeid'], how='inner')

    return mc5


def tcpl_fit2_nest(dat):
    modelnames = dat["model"].unique()

    dicts = {}
    for m in modelnames:
        ok = dat[dat["model"] == m].groupby("model_param")["model_val"].apply(float)  # throws warning,
        # tup = tuple(ok)

        # p = dict(tup)
        dicts[m] = ok.to_dict()

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
        else:
            modpars = None

        dicts[m]["pars"] = {x: dicts[m][x] for x in modpars}

    out = dicts
    return out
