import pandas as pd

from .query_db import query_db


def tcpl_output(dat, aeid):
    dat = pd.merge(tcpl_load_aeid(aeid), dat, on="aeid", how="right")
    dat = pd.merge(dat, tcpl_load_unit(aeid), on="aeid")
    dat = pd.merge(tcpl_load_chem("spid", dat["spid"].unique()), dat, on="spid", how="right")
    dat = pd.merge(dat, tcpl_load_conc_unit(dat["spid"].unique()), on="spid", how="left")

    # Subset chemicals with multiple samples!
    def subset_chemicals(dat):
        # A chid can have more than one spid. Use consensus hit (chit) logic: mean
        grouped = dat.groupby(["chid"])
        dat["chit"] = grouped["hitcall"].transform("mean")
        dat = dat.drop_duplicates(subset=["chid"], keep="first")
        return dat

    dat = subset_chemicals(dat)
    return dat


def tcpl_load_conc_unit(spid):
    qformat = """
        SELECT spid, tested_conc_unit AS conc_unit
        FROM sample
        WHERE spid IN ("{}");
        """

    spid_str = '","'.join(str(id) for id in spid)
    qstring = qformat.format(spid_str)
    dat = query_db(query=qstring)
    return dat


def tcpl_load_unit(aeid):
    qstring = f"""
        SELECT aeid, normalized_data_type AS resp_unit
        FROM assay_component_endpoint
        WHERE aeid = {aeid};
        """
    dat = query_db(query=qstring)
    return dat


def tcpl_load_chem(field=None, val=None, exact=True):
    qstring = """
      SELECT spid, chemical.chid, casn, chnm, dsstox_substance_id
      FROM sample
      LEFT JOIN chemical ON chemical.chid = sample.chid
      """
    qstring += " WHERE {0} IN ({1})".format(field, ", ".join("\"" + v + "\"" for v in val))
    dat = query_db(query=qstring)
    dat["code"] = "C" + dat["casn"].str.replace("-|_", "").fillna("")
    dat["chid"] = dat["chid"].astype(int)
    dat = dat.drop_duplicates()
    return dat


def tcpl_load_aeid(aeid):
    qstring = f"""
        SELECT assay_component_endpoint_name AS aenm, aeid 
        FROM assay_component_endpoint 
        WHERE aeid = {aeid};
        """
    dat = query_db(query=qstring)
    return dat
