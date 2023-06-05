import re

from query_db import tcpl_query
from tcplLoadData import prep_field


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

    fld = prep_field(fld=fld, tbl=tblo)
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
