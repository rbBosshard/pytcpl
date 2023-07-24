from query_db import tcpl_query

mc4_name = "mc4_"
mc4_agg_name = "mc4_agg_"
mc4_param_name = "mc4_param_"
mc5_name = "mc5_"
mc5_param_name = "mc5_param_"


def tcpl_list_flds(tbl):
    qformat = "DESCRIBE {tbl};"
    query = tcpl_query(qformat.format(tbl=tbl))
    return query["Field"].tolist()


def tcpl_load_data(lvl, fld, ids):
    if lvl == 3:
        tbls = ["mc0", "mc1", "mc3"]
        select_cols = ["mc3.m0id", "mc3.m1id", "mc3.m2id", "m3id", "spid", "aeid", "logc", "resp", "cndx", "wllt"]
        qformat = f"SELECT {', '.join(select_cols)} " \
                  f"FROM mc0, mc1, mc3 " \
                  f"WHERE mc0.m0id = mc1.m0id AND mc1.m0id = mc3.m0id"

    elif lvl == "agg":
        tbls = ["mc3", f"{mc4_agg_name}", f"{mc4_name}"]
        select_cols = [
            f"{mc4_agg_name}.aeid",
            f"{mc4_agg_name}.m4id",
            f"{mc4_agg_name}.m3id",
            f"{mc4_agg_name}.m2id",
            f"{mc4_agg_name}.m1id",
            f"{mc4_agg_name}.m0id",
            f"{mc4_name}.spid",
            "logc",
            "resp"]

        qformat = f"SELECT {', '.join(select_cols)} " \
                  f"FROM mc3, {mc4_agg_name}, {mc4_name} " \
                  f"WHERE mc3.m3id = {mc4_agg_name}.m3id " \
                  f"AND {mc4_name}.m4id = {mc4_agg_name}.m4id"

    elif lvl == 4:
        tbls = [f"{mc4_name}", f"{mc4_param_name}"]
        select_cols = [f"{mc4_name}.m4id", f"{mc4_name}.aeid", "spid", "bmad",
                       "model", "model_param", "model_val"]
        qformat = f"SELECT {', '.join(select_cols)} " \
                  f"FROM {mc4_param_name}, {mc4_name} " \
                  f"WHERE {mc4_name}.m4id = {mc4_param_name}.m4id"

    elif lvl == 5:
        tbls = [f"{mc4_name}", f"{mc5_name}", f"{mc5_param_name}"]
        select_cols = [f"{mc5_name}.m5id", f"{mc5_name}.m4id", f"{mc5_name}.aeid",
            "spid", "bmad", "hitc", "modl", "coff", "hit_param", "hit_val"]

        qformat = f"SELECT {', '.join(select_cols)} " \
                  f"FROM {mc4_name}, {mc5_name}, {mc5_param_name} " \
                  f"WHERE {mc4_name}.m4id = {mc5_name}.m4id " \
                  f"AND {mc5_name}.m5id = {mc5_param_name}.m5id"

    elif lvl == 6:
        tbls = [f"{mc5_name}", f"{mc5_param_name}"]
        select_cols = ["m4id", f"{mc5_param_name}.*", "modl", "coff"]
        qformat = f"SELECT {', '.join(select_cols)} " \
                  f"FROM {mc5_name} " \
                  f"JOIN {mc5_param_name} " \
                  f"ON {mc5_name}.m5id = {mc5_param_name}.m5id"
    else:
        raise ValueError("lvl not supported")

    def prep_field(fld, tbls):
        # scope with the fully qualified table.field name to avoid ambiguity
        for table in tbls:
            if fld in tcpl_list_flds(table):
                return f"{table}.{fld}"

    fld = prep_field(fld, tbls)
    ids = ids if isinstance(ids, list) else [ids]
    qstring = f"{qformat} AND {fld} IN ({','.join(map(str, ids))});"
    dat = tcpl_query(query=qstring)
    return dat
