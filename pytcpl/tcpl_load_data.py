from query_db import tcpl_query

mc4_name = "mc4_"
mc4_agg_name = "mc4_agg_"
mc4_param_name = "mc4_param_"
mc5_name = "mc5_"
mc5_param_name = "mc5_param_"


def tcpl_load_data(lvl, fld, ids):
    tbls = ["mc0", "mc1", "mc3"]
    select_cols = ["mc3.m0id", "mc3.m1id", "mc3.m2id", "m3id", "spid", "aeid", "logc", "resp", "cndx", "wllt"]
    qformat = f"SELECT {', '.join(select_cols)} " \
              f"FROM mc0, mc1, mc3 " \
              f"WHERE mc0.m0id = mc1.m0id AND mc1.m0id = mc3.m0id"

    def prep_field(fld, tbls):
        def tcpl_list_flds(tbl):
            qformat = "DESCRIBE {tbl};"
            query = tcpl_query(qformat.format(tbl=tbl))
            return query["Field"].tolist()

        # scope with the fully qualified table.field name to avoid ambiguity
        for table in tbls:
            if fld in tcpl_list_flds(table):
                return f"{table}.{fld}"

    fld = prep_field(fld, tbls)
    ids = ids if isinstance(ids, list) else [ids]
    qstring = f"{qformat} AND {fld} IN ({','.join(map(str, ids))});"
    dat = tcpl_query(query=qstring)
    return dat
