import pandas as pd

from query_db import tcpl_query
from tcplLoadAeid import tcpl_load_aeid
from tcplLoadUnit import tcpl_load_unit
from tcplLoadChem import tcpl_load_chem


def tcpl_prep_otpt(dat, ids=None):
    if not isinstance(dat, pd.DataFrame):
        raise ValueError("'dat' must be a DataFrame.")
        
    dnames = dat.columns.tolist()
    
    if ids is None:
        ids = dnames
            
    if "aeid" in ids:  # Map aeid names and resp_units
        if "aeid" not in dnames:
            print("Warning: 'aeid' field is not in dat. No 'aeid' mapping performed.")
        else:
            if "aenm" in dnames:
                dat = dat.drop("aenm", axis=1)
            if "resp_unit" in dnames:
                dat = dat.drop("resp_unit", axis=1)
            aeid_mapping = tcpl_load_aeid("aeid", dat["aeid"].unique())
            dat = pd.merge(aeid_mapping, dat, on="aeid", how="right")
            unit_mapping = tcpl_load_unit(dat["aeid"].unique())
            dat = pd.merge(dat, unit_mapping, on="aeid")
            
    if "spid" in ids:
        if "spid" not in dnames:
            print("Warning: 'spid' field is not in dat. No 'spid' mapping performed.")
        else:
            if "chid" in dnames:
                dat = dat.drop("chid", axis=1)
            if "casn" in dnames:
                dat = dat.drop("casn", axis=1)
            if "chnm" in dnames:
                dat = dat.drop("chnm", axis=1)
            if "code" in dnames:
                dat = dat.drop("code", axis=1)
            spid_mapping = tcpl_load_chem("spid", dat["spid"].unique())
            dat = pd.merge(spid_mapping, dat, on="spid", how="right")
            # Add conc units
            conc_unit_mapping = tcpl_load_conc_unit(dat["spid"].unique())
            dat = pd.merge(dat, conc_unit_mapping, on="spid", how="left")
            
    return dat


def tcpl_load_conc_unit(spid):
    qformat = """
    SELECT
      spid,
      tested_conc_unit AS conc_unit
    FROM
      sample
    WHERE
      spid IN ("{}");
    """

    spid_str = '","'.join(str(id) for id in spid)
    qstring = qformat.format(spid_str)

    dat = tcpl_query(query=qstring)

    if dat.shape[0] == 0:
        # print("The given spid(s) do not have concentration units.")
        return dat

    len_miss = sum(spid not in dat["spid"].values for spid in spid)
    if len_miss > 0:
        print(f"{len_miss} of the given spid(s) do not have concentration units.")

    return dat
