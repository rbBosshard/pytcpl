import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pytcpl.fit_models import get_params
from pytcpl.tcpl_hit_core import tcpl_hit_core
from pytcpl.tcpl_load_data import tcpl_load_data


def tcpl_hit(mc4, coff, parallelize=False, verbose=False):
    df = mc4[mc4['model'] != 'all']

    def tcpl_fit_nest(dat):
        modelnames = dat["model"].unique()
        dicts = {}
        for m in modelnames:
            df = dat[dat["model"] == m].groupby("model_param")["model_val"].apply(lambda x: float(x.iloc[0]))
            dicts[m] = df.to_dict()
            dicts[m]["pars"] = {x: dicts[m][x] for x in get_params(m)}
        return dicts

    if parallelize:  # Parallel: Split the DataFrame into groups and apply the function in parallel
        def apply_tcpl_fit_nest(name, group):
            out = tcpl_fit_nest(group[['model', 'model_param', 'model_val']])
            return pd.DataFrame({"m4id": [name], "params": [out]})

        nested_mc4 = Parallel(n_jobs=-1)(delayed(apply_tcpl_fit_nest)(name, group)
                                         for name, group in df.groupby('m4id'))
        nested_mc4 = pd.concat(nested_mc4).reset_index(drop=True)  # Concatenate the results into a single DataFrame

    else:  # Serial: For debugging
        nested_mc4 = df.groupby('m4id').apply(lambda x: tcpl_fit_nest(
            pd.DataFrame({'model': x['model'], 'model_param': x['model_param'],
                          'model_val': x['model_val']}))).reset_index(name='params')

    val = list(nested_mc4['m4id'].values)
    l4_agg = tcpl_load_data(lvl='agg', fld='m4id', val=val)
    # Might be a huge list: Maybe use BETWEEN sql operator to get all values between two values
    ids = list(l4_agg['m3id'].values)
    # heavy operation if ids list is long, aggregate ids from 3 huge tables
    # Create an empty list to store the DataFrames
    df_list = []
    # Iterate through the loop
    chunk_size = 10000
    num_chunks = len(ids) // chunk_size
    remaining_elements = len(ids) % chunk_size
    for i in range(num_chunks):
        # Create or obtain a DataFrame
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        df = tcpl_load_data(lvl=3, fld='m3id', val=ids[start_index:end_index])
        df_list.append(df)

    # Handle the last chunk
    if remaining_elements > 0:
        start_index = num_chunks * chunk_size
        end_index = start_index + remaining_elements
        df = tcpl_load_data(lvl=3, fld='m3id', val=ids[start_index:end_index])
        df_list.append(df)

    # Concatenate all the DataFrames in the list
    data = pd.concat(df_list)
    # data = tcpl_load_data(lvl=3, fld='m3id', val=val)

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

    def apply_tcpl_hit_core(params, conc, resp, bmed, cutoff, onesd):
        return tcpl_hit_core(params=params, conc=conc, resp=resp, bmed=bmed, cutoff=cutoff, onesd=onesd)

    if parallelize:
        # Parallelize the computation
        test = Parallel(n_jobs=-1)(
            delayed(tcpl_hit_core)(
                params=row['params'], conc=np.array(row.conc), resp=np.array(row.resp),
                bmed=row['bmed'], cutoff=coff, onesd=row['onesd']
            ) for _, row in nested_mc4.iterrows()
        )
        res = nested_mc4
        res = pd.concat([nested_mc4, pd.DataFrame(test)], axis=1)
    else:
        test = (
            nested_mc4.assign(df=lambda x: [
                tcpl_hit_core(params=x.params, conc=np.array(x.conc), resp=np.array(x.resp), bmed=x.bmed, cutoff=coff,
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
            (res['fit_model'] == 'none')
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
    mc5 = mc5[['m4id', 'aeid', 'fit_model', 'hitcall', 'fitc', 'cutoff']].rename(
        columns={"fit_model": "modl", "hitcall": "hitc", "cutoff": "coff"}).assign(model_type=2)

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
