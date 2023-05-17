import pandas as pd
import numpy as np

from tcplAppend import tcplAppend
from tcplQuery import tcplQuery
from tcplLoadData import tcplLoadData

def write_lvl_4(dat):
    mc4_cols = ["aeid", "spid", "bmad", "resp_max", "resp_min", "max_mean", "max_mean_conc", "max_med", "max_med_conc", "logc_max", "logc_min", "nconc", "npts", "nrep", "nmed_gtbl", "tmpi"]
    mc4_agg_cols = ["m" + str(i) + "id" for i in range(5)] + ["aeid"]

    tcplAppend(dat[mc4_cols].drop_duplicates(), "mc4_")

    qformat = "SELECT m4id, aeid, tmpi FROM mc4 WHERE aeid IN ({})"
    ids = dat["aeid"].unique()
    qstring = qformat.format(",".join(["'" + str(i) + "'" for i in ids]))
    m4id_map = tcplQuery(query=qstring)

    m4id_map.set_index(["aeid", "tmpi"], inplace=True)
    dat.set_index(["aeid", "tmpi"], inplace=True)
    dat = dat.loc[m4id_map.index]
    param = dat[["m4id", "aeid", "fitparams"]]

    #get one standard deviation to save in similar way to fit params
    onesd = dat[["m4id", "aeid", "osd"]].rename(columns={"osd": "model_val"})
    onesd["model"] = "all"
    onesd["model_param"] = "onesd"
    if "bmed" in dat.columns:
        bmed = dat[["m4id", "aeid", "bmed"]].rename(columns={"bmed": "model_val"})
        bmed["model"] = "all"
        bmed["model_param"] = "bmed"
    else:
        bmed = pd.DataFrame()
    
    #unnest fit2 params
    unnested_param = pd.concat([pd.DataFrame(tcplFit2_unnest(x)) for x in param['fitparams']], keys=param['m4id'], names=['m4id']).reset_index()
    unnested_param['m4id'] = pd.to_numeric(unnested_param['m4id'])

    unnested_param.set_index("m4id", inplace=True)
    param.set_index("m4id", inplace=True)
    dat1 = param.loc[unnested_param.index]
    dat_param = dat1[["aeid", "model", "model_param", "model_val"]].reset_index()

    # get l3 dat for agg columns
    dat_agg = dat[['aeid', 'm4id']].assign(m3id=dat['m3ids']).groupby("m4id").apply(lambda x: x.apply(pd.Series.explode))
    l3_dat = tcplLoadData(lvl = 3, fld = "m3id", val = dat_agg['m3id'], type = "mc")["m0id","m1id","m2id","m3id"]
    dat_agg.set_index("m3id", inplace=True)
    l3_dat.set_index("m3id", inplace=True)
    dat_agg = dat_agg.loc[l3_dat.index]

    print(f'dat_agg: {dat_agg}')
    print(f'dat_param: {dat_param}')
    print(f'onesd: {onesd}')
    print(f'bmed: {bmed}')


    # tcplAppend(dat_agg[mc4_agg_cols], "mc4_agg_")

    # tcplAppend(dat_param, "mc4_param_")

    # tcplAppend(onesd, "mc4_param_")

    # tcplAppend(bmed, "mc4_param_")


def tcplFit2_unnest(output):
    modelnames = output['modelnames']
    for m in modelnames:
        globals()[m] = output[m][~output[m].columns.isin(["pars", "sds", "modl"])]
    res = {m: globals()[m] for m in modelnames}
    test = pd.DataFrame(columns=['model', 'model_param', 'model_val'])
    for m in modelnames:
        lst = res[m].apply(lambda x: np.nan if len(x) < 1 else x, axis=0)
        test = test.append(pd.DataFrame({'model': m, 'model_param': lst.index, 'model_val': lst.values}), ignore_index=True)
    return test

def tcplFit2_nest(dat):
    modelnames = dat['model'].unique()
    for m in modelnames:
        globals()[m] = dat[dat['model'] == m]['model_val'].groupby(dat[dat['model'] == m]['model_param']).apply(list).to_dict()
    for m in modelnames:
        if m == "cnst":
            modpars = ["er"]
        elif m == "exp2":
            modpars = ["a", "b", "er"]
        globals()[m]['pars'] = modpars
    out1 = {m: globals()[m] for m in modelnames}
    out1['modelnames'] = modelnames
    return out1

def log_model_params(datc):
    datc['model_val'] = np.where(datc['model_param'] == 'ac50', np.log10(datc['model_val']), datc['model_val'])
    return datc

def unlog_model_params(datc):
    datc['model_val'] = np.where(datc['model_param'] == 'ac50', 10**(datc['model_val']), datc['model_val'])
    return datc
