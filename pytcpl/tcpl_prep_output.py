import re

import pandas as pd

from query_db import tcpl_query
from tcpl_load_data import prep_field


def tcpl_prep_output(dat, ids=None):
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


def tcpl_load_unit(aeid):
    qformat = """
    SELECT
      aeid,
      normalized_data_type AS resp_unit
    FROM
      assay_component_endpoint
    WHERE
      aeid IN ({});
    """

    aeid_str = ",".join(str(id) for id in aeid)
    qstring = qformat.format(aeid_str)

    dat = tcpl_query(query=qstring)

    if dat.shape[0] == 0:
        print("Warning: The given aeid(s) do not have response units.")
        return dat

    len_miss = sum(aeid not in dat["aeid"].values for aeid in aeid)
    if len_miss > 0:
        print(f"Warning: {len_miss} of the given aeid(s) do not have response units.")

    return dat


def tcpl_load_chem(field=None, val=None, exact=True, include_spid=True):
    field_mapping = {
        "chid": "chid",
        "spid": "spid",
        "chnm": "chnm",
        "casn": "casn",
        "code": "code",
        "chem.only": "chem.only",
        "dsstox_substance_id": "dsstox_substance_id"
    }

    if field is not None:
        if field not in field_mapping:
            raise ValueError("Invalid 'field' value.")

    qstring = _chem_q(field=field, val=val, exact=exact)

    dat = tcpl_query(query=qstring)

    if dat.shape[0] == 0:
        print("Warning: The given {}(s) are not in the tcpl database.".format(field))
        return dat

    dat["code"] = None
    dat.loc[~dat["casn"].isna(), "code"] = "C" + dat["casn"].str.replace("-|_", "")

    dat["chid"] = dat["chid"].astype(int)
    dat = dat.drop_duplicates()

    if include_spid:
        return dat

    dat = dat[["chid", "chnm", "casn", "code", "dsstox_substance_id"]]
    return dat


def _chem_q(field, val, exact):
    qstring = """
      SELECT spid, chemical.chid, casn, chnm, dsstox_substance_id
      FROM sample
      LEFT JOIN chemical ON chemical.chid = sample.chid
      """

    if field is not None:
        nfld = {
            "spid": "spid",
            "chid": "chemical.chid",
            "casn": "casn",
            "code": "casn",
            "chem.only": "chem.only",
            "dsstox_substance_id": "dsstox_substance_id",
            "chnm": "chnm"
        }.get(field)

        # if field == "code":
        #     val = [tcplCode2CASN(v) for v in val]

        qstring += " WHERE"

        if nfld == "chnm":
            if exact:
                qstring += " chnm IN ({0});".format(", ".join("\"" + v + "\"" for v in val))
            else:
                qstring += " chnm RLIKE {0};".format("\"" + "|".join(val) + "\"")
        elif nfld == "chem.only":
            qstring = """
                SELECT *
                FROM chemical
                """
        else:
            qstring += " {0} IN ({1})".format(nfld, ", ".join("\"" + v + "\"" for v in val))

    return qstring


def tcpl_load_aeid(fld=None, val=None):
    out = ["assay_component_endpoint.aeid", "assay_component_endpoint.assay_component_endpoint_name"]
    qstring = build_assay_q(out=out, tblo=[0, 1, 3, 2, 5], fld=fld, val=val)
    dat = tcpl_query(query=qstring)
    return dat


def build_assay_q(out, tblo, fld=None, val=None):
    tbls = ["assay_source", "assay", "assay_component", "assay_component_map",
            "assay_component", "assay_component_endpoint"]

    tblo = [tbls[i] for i in tblo]

    fld = convert_names(fld)

    fld = prep_field(fld=fld, tbls=tblo)
    if not isinstance(fld, list):
        fld = [fld]
    # correct?
    afld = list(set(fld + out))
    atbl = list(set([x.split(".")[0] for x in afld]))
    tbls = list(set(atbl))
    if not any(["map" in field for field in afld]):
        tbls = [tbl for tbl in tbls if tbl != "assay_component_map"]

    if len(tbls) > 1:
        tbl_link = []
        if all(["assay_source" in tbls, "assay" in tbls]):
            tbl_link.append("assay_source.asid = assay.asid")
        if all(["assay" in tbls, "assay_component" in tbls]):
            tbl_link.append("assay.aid = assay_component.aid")
        if all(["assay_component" in tbls, "assay_component_map" in tbls]):
            tbl_link.append("assay_component.acid = assay_component_map.acid")
        if all(["assay_component" in tbls, "assay_component_endpoint" in tbls]):
            tbl_link.append("assay_component.acid = assay_component_endpoint.acid")
    else:
        tbl_link = []

    qformat = f"SELECT {', '.join(afld)} FROM {', '.join(tbls)}"
    if len(tbl_link) > 0:
        qformat += " WHERE " + " AND ".join(tbl_link)

    if len(fld) > 0:
        qformat += " AND " if len(tbl_link) > 0 else " WHERE "
        qformat += " AND ".join([f"{f} IN (%s)" for f in fld])
        qformat += ";"

        if not isinstance(val, list):
            val = [val]

        val = ','.join((str(v) for v in val)).replace("[", "").replace("]", "")

        qstring = qformat % val
    else:
        qstring = qformat

    qstring = re.sub("assay_source_name", "assay_source_name AS asnm", qstring)
    qstring = re.sub("assay_name", "assay_name AS anm", qstring)
    qstring = re.sub("assay_component_name", "assay_component_name AS acnm", qstring)
    qstring = re.sub("assay_component_endpoint_name", "assay_component_endpoint_name AS aenm", qstring)

    return qstring


def convert_names(name):
    name = re.sub("aenm", "assay_component_endpoint_name", name)
    name = re.sub("acnm", "assay_component_name", name)
    name = re.sub("anm", "assay_name", name)
    name = re.sub("asnm", "assay_source_name", name)
    return name
