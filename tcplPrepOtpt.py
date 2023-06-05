import pandas as pd

from tcplLoadAeid import tcplLoadAeid
from tcplLoadUnit import tcplLoadUnit
from tcplLoadChem import tcplLoadChem
from tcplLoadConcUnit import tcplLoadConcUnit

def tcplPrepOtpt(dat, ids=None):
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
            aeid_mapping = tcplLoadAeid("aeid", dat["aeid"].unique())  # Assuming tcplLoadAeid is a function that loads aeid mappings
            dat = pd.merge(aeid_mapping, dat, on="aeid", how="right")
            unit_mapping = tcplLoadUnit(dat["aeid"].unique())  # Assuming tcplLoadUnit is a function that loads unit mappings
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
            spid_mapping = tcplLoadChem("spid", dat["spid"].unique())  # Assuming tcplLoadChem is a function that loads spid mappings
            dat = pd.merge(spid_mapping, dat, on="spid", how="right")
            # Add conc units
            conc_unit_mapping = tcplLoadConcUnit(dat["spid"].unique())  # Assuming tcplLoadConcUnit is a function that loads conc unit mappings
            dat = pd.merge(dat, conc_unit_mapping, on="spid", how="left")
            
    return dat