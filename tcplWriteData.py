import pandas as pd
import numpy as np
import os

from tcplAppend import tcplAppend
from query_db import tcplQuery
from tcplCascade import tcplCascade
from write_lvl_4 import write_lvl_4

def tcplWriteData(dat, lvl):
    # Check for valid inputs
    if lvl not in [4, 5]:
        raise ValueError("Invalid lvl input - must be an integer 4 or 5.")
    
    fkey = "aeid" if lvl > 2 else "acid"
    ids = dat[fkey].unique()

    if len(ids) >= 500:
        ibins = np.array_split(ids, np.ceil(len(ids) / 500))
        for x in ibins:
            tcplCascade(lvl=lvl, id=x)
    else:
        tcplCascade(lvl=lvl, id=ids)

    dat["modified_by"] = os.getlogin()

    if lvl == 4:
        if "fitparams" in dat.columns:
            write_lvl_4(dat)
    elif lvl == 5:
        tcplAppend(dat=dat[["aeid", "m4id", "m5id", "hitc", "fitc", "coff", "model_type", "modified_by"]].drop_duplicates(), tbl="mc5_")
        # get m5id for mc5_param
        qformat = "SELECT m5id, m4id, aeid FROM mc5 WHERE aeid IN (%s);"
        qstring = qformat % ",".join('"' + str(id) + '"' for id in ids)

        m5id_map = tcplQuery(query=qstring)
        m5id_map = m5id_map.set_index(["aeid", "m4id"])
        dat = dat.set_index(["aeid", "m4id"])
        dat = dat.join(m5id_map, how="left")

        tcplAppend(dat=dat[["m5id", "aeid", "hit_param", "hit_val"]],tbl="mc5_param_")
    else:
        n = dat.shape[0]
        tbl = "mc" + str(lvl)
        if n <= 1e6:
            tcplAppend(dat=dat, tbl=tbl)
        else:
            rbins = np.array_split(np.arange(n), np.ceil(n / 1e6))
            for x in rbins:
                tcplAppend(dat=dat.iloc[x], tbl=tbl)

    return True
