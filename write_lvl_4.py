import pandas as pd
import numpy as np

from tcplAppend import tcplAppend
from query_db import tcplQuery
from tcplLoadData import tcplLoadData

def write_lvl_4(dat):
    
    mc4_cols = ["aeid", "spid", "bmad", "resp_max", "resp_min", "max_mean", "max_mean_conc", "max_med", "max_med_conc", "logc_max", "logc_min", "nconc", "npts", "nrep", "nmed_gtbl", "tmpi", "modified_by"]
    mc4_agg_cols = ["m" + str(i) + "id" for i in range(5)] + ["aeid"]
   
    tcplAppend(dat[mc4_cols].drop_duplicates(), "mc4_")

    qformat = "SELECT m4id, aeid, tmpi FROM mc4_ WHERE aeid IN ({})"
    ids = dat["aeid"].unique()
    qstring = qformat.format(",".join(["'" + str(i) + "'" for i in ids]))
    m4id_map = tcplQuery(query=qstring)

    m4id_map = m4id_map.set_index(["aeid", "tmpi"])
    dat = dat.set_index(["aeid", "tmpi"])
    dat = dat.join(m4id_map, how="left")
    dat = dat.reset_index()
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

    unnested_param = unnested_param.set_index("m4id")
    param = param.set_index("m4id")
    dat1 = param.join(unnested_param, how="left")
    dat_param = dat1[["aeid", "model", "model_param", "model_val"]].reset_index()

    # get l3 dat for agg columns
    dat_agg = dat[['aeid', 'm4id']].assign(m3id=dat['m3ids'])

    # how many m3ids are there per m4id datapoint?
    # for i in range(dat_agg.shape[0]):
    #     print(f'shape: {len(dat_agg.m3id.iloc[i])}')
    dat_agg = dat_agg.set_index('m4id')

    dat_agg = dat_agg.explode('m3id').reset_index()

    ids = list(dat_agg['m3id'])
    l3_dat = tcplLoadData(lvl = 3, fld = "m3id", val = ids, type = "mc")

    l3_dat = l3_dat[["m0id","m1id","m2id","m3id"]]
    dat_agg = dat_agg.set_index("m3id")
    l3_dat = l3_dat.set_index("m3id")
    dat_agg = dat_agg.join(l3_dat, how="left")
    dat_agg = dat_agg.reset_index()

    # print(f'dat_agg: {dat_agg}')
    # print(f'dat_param: {dat_param}')
    # print(f'onesd: {onesd}')
    # print(f'bmed: {bmed}')

    tcplAppend(dat_agg[mc4_agg_cols], "mc4_agg_")
    tcplAppend(dat_param, "mc4_param_")
    tcplAppend(onesd, "mc4_param_")
    tcplAppend(bmed, "mc4_param_")


def tcplFit2_unnest(output):
    modelnames = list(output.keys())

    res = {}

    for m in modelnames:
        res[m] = {name: val for name, val in output[m].items() if name not in ["pars", "sds", "modl"]}
        # pars = output[m]["pars"]
        # if pars is not np.nan:
        res[m].update(output[m]["pars"])

    test = pd.DataFrame(columns=["model", "model_param", "model_val"])
    for m in modelnames:
        # print([(m, param , val) for param, val in res[m].items()])
        lst = {param: val for param, val in res[m].items()}
        new_row = pd.DataFrame({"model": m, "model_param": list(lst.keys()), "model_val": list(lst.values())})
        test = pd.concat([test, new_row])

    return test


def log_model_params(datc):
    datc.loc[datc["model_param"] == "ac50", "model_val"] = np.log10(datc.loc[datc["model_param"] == "ac50", "model_val"])
    return datc


def unlog_model_params(datc):
    datc.loc[datc["model_param"] == "ac50", "model_val"] = 10 ** datc.loc[datc["model_param"] == "ac50", "model_val"]
    return datc
