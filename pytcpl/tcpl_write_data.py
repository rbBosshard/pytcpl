import ast
import os

import pandas as pd

from query_db import tcpl_query, get_sqlalchemy_engine
from tcpl_load_data import tcpl_load_data

mc4_name = "mc4_"
mc4_param_name = "mc4_param_"
mc4_agg_name = "mc4_agg_"
mc5_name = "mc5_"
mc5_param_name = "mc5_param_"


def tcpl_fit_unnest(output):
    try:
        output = ast.literal_eval(output) if isinstance(output, str) else output  # Either from csv or db
    except Exception as e:
        print(e)
    model_names = list(output.keys())
    res = {}
    for m in model_names:
        d = output[m]
        res[m] = {name: val for name, val in d.items() if name not in ["pars", "sds", "modl"]}
        res[m].update(d["pars"])

    rows = [{"model": m, "model_param": p, "model_val": val} for m in model_names for p, val in
            res[m].items()]

    out = pd.DataFrame.from_records(rows, columns=["model", "model_param", "model_val"])

    return out



def tcpl_write_data(id, dat, lvl):
    if lvl not in [4, 5]:
        raise ValueError("Invalid lvl input - must be an integer 4 or 5.")

    # Delete old data in cascade
    if lvl <= 4:
        tcpl_delete(tbl=mc4_name, id=id)
        tcpl_delete(tbl=mc4_agg_name, id=id)
        tcpl_delete(tbl=mc4_param_name, id=id)
    if lvl <= 5:
        tcpl_delete(tbl=mc5_name, id=id)
        tcpl_delete(tbl=mc5_param_name, id=id)

    # dat["modified_by"] = os.getlogin()

    if lvl == 4:
        mc4_cols_slim = ["aeid", "spid", "bmad"]
        mc4_agg_cols = ["m" + str(i) + "id" for i in range(5)] + ["aeid"]
        tcpl_append(dat[mc4_cols_slim], mc4_name)

        qstring = f"SELECT m4id, aeid FROM {mc4_name} WHERE aeid IN ({id})"
        m4id_map = tcpl_query(query=qstring)
        dat = pd.concat([dat, m4id_map], axis=1)  # TODO: recheck, replaced tmpi
        # Drop duplicated columns, keeping the first occurrence
        dat = dat.loc[:, ~dat.columns.duplicated()]

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

        # unnest fit params
        unnested_param = pd.concat([pd.DataFrame(tcpl_fit_unnest(x)) for x in param['fitparams']], keys=param['m4id'],
                                   names=['m4id']).reset_index()

        unnested_param = unnested_param.set_index("m4id")
        param = param.set_index("m4id")
        dat1 = param.join(unnested_param, how="left")
        dat_param = dat1[["aeid", "model", "model_param", "model_val"]].reset_index()

        tcpl_append(dat_param, mc4_param_name)
        tcpl_append(onesd, mc4_param_name)
        tcpl_append(bmed, mc4_param_name)

        # get l3 dat for agg columns
        m3id_ = dat['m3ids']
        if isinstance(m3id_.iloc[0], str):
            m3id_ = m3id_.apply(lambda x: ast.literal_eval(x))
        dat_agg = dat[['aeid', 'm4id']].assign(m3id=m3id_)
        dat_agg = dat_agg.set_index('m4id')
        dat_agg = dat_agg.explode('m3id')
        dat_agg = dat_agg.reset_index()

        ids = list(dat_agg['m3id'])
        df_list = []
        chunk_size = 10000
        num_chunks = len(ids) // chunk_size
        remaining_elements = len(ids) % chunk_size
        for i in range(num_chunks):
            # Create or obtain a DataFrame
            start_index = i * chunk_size
            end_index = start_index + chunk_size
            df = tcpl_load_data(lvl=3, fld="m3id", ids=ids[start_index:end_index])
            df_list.append(df)

        # Handle the last chunk
        if remaining_elements > 0:
            start_index = num_chunks * chunk_size
            end_index = start_index + remaining_elements
            df = tcpl_load_data(lvl=3, fld="m3id", ids=ids[start_index:end_index])
            df_list.append(df)

        # Concatenate all the DataFrames in the list
        l3_dat = pd.concat(df_list)

        l3_dat = l3_dat[["m0id", "m1id", "m2id", "m3id"]]
        dat_agg = dat_agg.set_index("m3id")
        l3_dat = l3_dat.set_index("m3id")
        dat_agg = dat_agg.join(l3_dat, how="left")
        dat_agg = dat_agg.reset_index()
        tcpl_append(dat_agg[mc4_agg_cols], mc4_agg_name)

    elif lvl == 5:
        mc5_cols_slim = ["m4id", "aeid", "modl", "hitc", "coff"]
        tcpl_append(dat=dat[mc5_cols_slim].drop_duplicates(), tbl=mc5_name)
        qstring = f"SELECT m5id, m4id, aeid FROM {mc5_name} WHERE aeid IN ({id});"
        m5id_map = tcpl_query(query=qstring)
        m5id_map = m5id_map.set_index(["aeid", "m4id"])
        dat = dat.set_index(["aeid", "m4id"])
        dat = dat.join(m5id_map, how="left").reset_index()
        tcpl_append(dat=dat[["m5id", "aeid", "hit_param", "hit_val"]], tbl=mc5_param_name)

def process_and_write_chunk(chunk, tbl):
    engine = get_sqlalchemy_engine()
    try:
        chunk.to_sql(tbl, engine, if_exists='append', index=False)
    except Exception as err:
        print(err)

def tcpl_append(dat, tbl):
    try:
        # if(len(dat) < 1000):
        engine = get_sqlalchemy_engine()
        dat.to_sql(tbl, engine, if_exists='append', index=False)
        # else:
        #     chunk_size = 5000  # 5000 best
        #     chunks = [dat[i:i + chunk_size] for i in range(0, len(dat), chunk_size)]
        #     # num_rows_affected = Parallel(n_jobs=-1)(delayed(process_and_write_chunk)(chunk, tbl) for chunk in chunks)
        #     num_rows_affected = [process_and_write_chunk(chunk, tbl) for chunk in chunks]
    except Exception as err:
        print(err)


def tcpl_delete(tbl, id):
    qstring = f"DELETE FROM {tbl} WHERE aeid = {id};"
    tcpl_query(qstring)
