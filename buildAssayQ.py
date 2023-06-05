import re
import yaml
from tcplLoadData import prepField
from convertNames import convertNames


def buildAssayQ(out, tblo, fld=None, val=None, add_fld=None):
    tbls = ["assay_source", "assay", "assay_component", "assay_component_map",
            "assay_component", "assay_component_endpoint"]

    tblo = [tbls[i] for i in tblo]

    fld = convertNames(fld)
    # add_fld = convertNames(add_fld)

    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        db = config['DATABASE']['DB']

    fld = prepField(fld=fld, tbl=tblo, db=db)
    if not isinstance(fld, list):
        fld = [fld] 
    # add_fld = prepField(fld=add_fld, tbl=tblo, db=db)
    # afld = list(set(fld + out + add_fld))
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
