from tcpl_prep_otpt import tcpl_prep_otpt


def tcpl_subset_chid(dat, flag=False):
    if "m5id" not in dat.columns:
        raise ValueError("'dat' must be a DataFrame with level 5 data.")
    if "casn" not in dat.columns:
        dat = tcpl_prep_otpt(dat)

    dat["hitc"] = dat["hitc"] >= 0.9
    dat["chit"] = dat.groupby(["aeid", "chid"])["hitc"].transform(lambda x: x.mean() >= 0.5)
    dat = dat[dat["hitc"] == dat["chit"] | (dat["chit"].isna() & (dat["hitc"] == -1 | dat["m4id"].isna()))]

    dat["fitc_ordr"] = None
    dat.loc[dat["fitc"].isin([37, 41, 46, 50]), "fitc_ordr"] = 0
    dat.loc[dat["fitc"].isin([38, 42, 47, 51]), "fitc_ordr"] = 1
    dat.loc[dat["fitc"].isin([36, 40, 45, 49]), "fitc_ordr"] = 2

    if flag:
        pass
    else:
        dat["nflg"] = False

    # also add "ac50", 
    dat = dat.sort_values(by=["aeid", "chid", "fitc_ordr", "nflg", "max_med"],
                          ascending=[True, True, True, True, False], na_position="last")

    min_modl_ga = dat.groupby(["aeid", "casn"]).apply(lambda x: x.index[0]).reset_index(name="ind")

    dat = dat.loc[min_modl_ga["ind"]]

    return dat
