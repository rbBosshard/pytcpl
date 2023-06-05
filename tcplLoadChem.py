import pandas as pd
from query_db import tcplQuery

def tcplLoadChem(field=None, val=None, exact=True, include_spid=True):
    tbl = ["chemical", "sample"]
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

    qstring = _ChemQ(field=field, val=val, exact=exact)

    dat = tcplQuery(query=qstring)
    dat = pd.DataFrame(dat)

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


def _ChemQ(field, val, exact):
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