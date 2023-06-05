import pandas as pd
from tcplPrepOtpt import tcplPrepOtpt

def tcplSubsetChid(dat, flag=False):
    if "m5id" not in dat.columns:
        raise ValueError("'dat' must be a DataFrame with level 5 data.")
    if "casn" not in dat.columns:
        dat = tcplPrepOtpt(dat)  # Assuming tcplPrepOtpt is a function that preprocesses the data

    dat["hitc"] = dat["hitc"] >= 0.9
    dat["chit"] = dat.groupby(["aeid", "chid"])["hitc"].transform(lambda x: x.mean() >= 0.5)
    dat = dat[dat["hitc"] == dat["chit"] | (dat["chit"].isna() & (dat["hitc"] == -1 | dat["m4id"].isna()))]

    dat["fitc.ordr"] = None
    dat.loc[dat["fitc"].isin([37, 41, 46, 50]), "fitc.ordr"] = 0
    dat.loc[dat["fitc"].isin([38, 42, 47, 51]), "fitc.ordr"] = 1
    dat.loc[dat["fitc"].isin([36, 40, 45, 49]), "fitc.ordr"] = 2

    if flag:
        pass
        # tst = isinstance(flag, bool)
        # if tst:
        #     prs = {}
        # else:
        #     prs = {"fld": "mc6_mthd_id", "val": flag}
        # flg = tcplLoadData(lvl=6, prs)
        # flg = flg.groupby("m4id").size().reset_index(name="nflg")
        # dat = pd.merge(dat, flg, on="m4id", how="left")
        # dat["nflg"] = dat["nflg"].fillna(0)
    else:
        dat["nflg"] = False
    
    # also add "ac50", 
    dat = dat.sort_values(by=["aeid", "chid", "fitc.ordr", "nflg", "max_med"], ascending=[True, True, True, True, False], na_position="last")
    
    min_modl_ga = dat.groupby(["aeid", "casn"]).apply(lambda x: x.index[0]).reset_index(name="ind")

    dat = dat.loc[min_modl_ga["ind"]]
    
    return dat
