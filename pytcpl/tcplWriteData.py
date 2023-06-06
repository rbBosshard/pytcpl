import time
import pandas as pd
import numpy as np
import os

from query_db import tcpl_query, get_sqlalchemy_engine
from tcplLoadData import tcpl_load_data


def tcpl_write_data(dat, lvl, verbose):
    # Check for valid inputs
    if lvl not in [4, 5]:
        raise ValueError("Invalid lvl input - must be an integer 4 or 5.")

    fkey = "aeid" if lvl > 2 else "acid"
    ids = dat[fkey].unique()

    if len(ids) >= 500:
        ibins = np.array_split(ids, np.ceil(len(ids) / 500))
        for x in ibins:
            tcpl_cascade(lvl=lvl, id=x, verbose=False)
    else:
        tcpl_cascade(lvl=lvl, id=ids, verbose=False)

    dat["modified_by"] = os.getlogin()

    if lvl == 4:
        mc4_cols = ["aeid", "spid", "bmad", "resp_max", "resp_min", "max_mean", "max_mean_conc", "max_med",
                    "max_med_conc", "logc_max", "logc_min", "nconc", "npts", "nrep", "nmed_gtbl", "tmpi", "modified_by"]
        mc4_agg_cols = ["m" + str(i) + "id" for i in range(5)] + ["aeid"]
        tcpl_append(dat[mc4_cols], "mc4_", False)

        qformat = "SELECT m4id, aeid, tmpi FROM mc4_ WHERE aeid IN ({})"
        ids = dat["aeid"].unique()
        qstring = qformat.format(",".join(["'" + str(i) + "'" for i in ids]))
        m4id_map = tcpl_query(query=qstring, verbose=False)

        m4id_map = m4id_map.set_index(["aeid", "tmpi"])
        dat = dat.set_index(["aeid", "tmpi"])
        dat = dat.join(m4id_map, how="left")
        dat.reset_index(inplace=True)
        param = dat[["m4id", "aeid", "fitparams"]]

        # get one standard deviation to save in similar way to fit params
        onesd = dat[["m4id", "aeid", "osd"]].rename(columns={"osd": "model_val"})
        onesd["model"] = "all"
        onesd["model_param"] = "onesd"
        if "bmed" in dat.columns:
            bmed = dat[["m4id", "aeid", "bmed"]].rename(columns={"bmed": "model_val"})
            bmed["model"] = "all"
            bmed["model_param"] = "bmed"
        else:
            bmed = pd.DataFrame()

        def tcpl_fit2_unnest(output):
            modelnames = list(output.keys())
            res = {}
            for m in modelnames:
                res[m] = {name: val for name, val in output[m].items() if name not in ["pars", "sds", "modl"]}
                res[m].update(output[m]["pars"])

            rows = [{"model": m, "model_param": p, "model_val": val} for m in modelnames for p, val in
                    res[m].items()]
            return pd.DataFrame.from_records(rows, columns=["model", "model_param", "model_val"])

        # unnest fit2 params
        unnested_param = pd.concat([pd.DataFrame(tcpl_fit2_unnest(x)) for x in param['fitparams']], keys=param['m4id'],
                                   names=['m4id']).reset_index()
        unnested_param = unnested_param.set_index("m4id")
        param = param.set_index("m4id")
        dat1 = param.join(unnested_param, how="left")
        dat_param = dat1[["aeid", "model", "model_param", "model_val"]].reset_index()
        tcpl_append(dat_param, "mc4_param_", False)
        tcpl_append(onesd, "mc4_param_", False)
        tcpl_append(bmed, "mc4_param_", False)

        # get l3 dat for agg columns
        dat_agg = dat[['aeid', 'm4id']].assign(m3id=dat['m3ids'])
        dat_agg = dat_agg.set_index('m4id')
        dat_agg = dat_agg.explode('m3id')
        dat_agg = dat_agg.reset_index()

        ids = list(dat_agg['m3id'])
        l3_dat = tcpl_load_data(lvl=3, fld="m3id",
                                val=ids)  # heavy operation if ids list is long, aggregate ids from 3 huge tables
        l3_dat = l3_dat[["m0id", "m1id", "m2id", "m3id"]]
        dat_agg = dat_agg.set_index("m3id")
        l3_dat = l3_dat.set_index("m3id")
        dat_agg = dat_agg.join(l3_dat, how="left")
        dat_agg = dat_agg.reset_index()
        tcpl_append(dat_agg[mc4_agg_cols], "mc4_agg_", False)  # lazy

    elif lvl == 5:
        mc5_name = "mc5_"
        tcpl_append(
            dat=dat[["m4id", "aeid", "modl", "hitc", "fitc", "coff", "model_type", "modified_by"]].drop_duplicates(),
            tbl=mc5_name, verbose=False)
        # get m5id for mc5_param
        qformat = f"SELECT m5id, m4id, aeid FROM {mc5_name} WHERE aeid IN (%s);"
        qstring = qformat % ",".join('"' + str(id) + '"' for id in ids)

        m5id_map = tcpl_query(query=qstring, verbose=False)
        m5id_map = m5id_map.set_index(["aeid", "m4id"])
        dat = dat.set_index(["aeid", "m4id"])
        dat = dat.join(m5id_map, how="left").reset_index()

        mc5_param_name = "mc5_param_"

        tcpl_append(dat=dat[["m5id", "aeid", "hit_param", "hit_val"]], tbl=mc5_param_name, verbose=False)


def log_model_params(datc):
    datc.loc[datc["model_param"] == "ac50", "model_val"] = np.log10(
        datc.loc[datc["model_param"] == "ac50", "model_val"])
    return datc


def unlog_model_params(datc):
    datc.loc[datc["model_param"] == "ac50", "model_val"] = 10 ** datc.loc[datc["model_param"] == "ac50", "model_val"]
    return datc


def tcpl_append(dat, tbl, verbose):
    engine = get_sqlalchemy_engine()
    with engine.begin() as connection:
        start_time = time.time()
        num_rows_affected = dat.to_sql(name=tbl, con=connection, if_exists="append", index=False)
        if verbose:
            print(f"Append to {tbl} >> {num_rows_affected} affected rows >> {str(time.time() - start_time)} seconds.")
    return num_rows_affected


def tcpl_cascade(lvl, id, verbose):
    if lvl <= 4:
        tcpl_delete(tbl="mc4_", fld="aeid", val=id, verbose=False)
    if lvl <= 4:
        tcpl_delete(tbl="mc4_agg_", fld="aeid", val=id, verbose=False)
    if lvl <= 4:
        tcpl_delete(tbl="mc4_param_", fld="aeid", val=id, verbose=False)
    if lvl <= 5:
        tcpl_delete(tbl="mc5_", fld="aeid", val=id, verbose=False)
    if lvl <= 5:
        tcpl_delete(tbl="mc5_param_", fld="aeid", val=id, verbose=False)
    if verbose:
        print("Completed delete cascade for", len(id), "ids\n")


def tcpl_delete(tbl, fld, val, verbose):
    qformat = f"DELETE FROM {tbl} WHERE"
    qformat += f" {' AND '.join([f'{fld} IN (%s)' for _ in val])}"
    qformat += ";"

    if not isinstance(val, list):
        val = [val]
    val = [','.join([f'"{x}"' for x in v]) for v in val]

    qstring = qformat % tuple(val)
    tcpl_query(qstring, False)
