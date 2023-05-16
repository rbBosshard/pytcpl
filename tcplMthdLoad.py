from tcplQuery import tcplQuery

def tcplMthdLoad(lvl, id = None, type = "mc"):
    if isinstance(id, int) or isinstance(id, str):
        id = list(map(str, [id]))
    id_name = "acid" if type == "mc" and lvl == 2 else "aeid"
    flds = [id_name, f"b.{type}{lvl}_mthd AS mthd", f"b.{type}{lvl}_mthd_id AS mthd_id"]
    if lvl < 4 and type == "mc":
        flds.append("a.exec_ordr AS ordr")
    tbls = [f"%s_{id_name} AS a", f"%s_methods AS b"]
    qformat = f"SELECT {', '.join(flds)} FROM {', '.join(tbls)} WHERE a.%s_mthd_id = b.%s_mthd_id"
    qformat = qformat.replace("%s", f"{type}{lvl}")
    if id is not None:
        qformat = f"{qformat} AND {id_name} IN ({', '.join(id)});"
    if (lvl < 4 and type == "mc") or (lvl == 1 and type == "sc"):
        qstring = f"{qformat} ORDER BY {id_name}, a.exec_ordr"
    else:
        qstring = qformat
    # if verbose:
    #     print(f"qstring: {qstring}")
    dat = tcplQuery(query=qstring)
    return dat