from query_db import query_db


def tcpl_mthd_load(lvl, aeid):
    flds = ["aeid", f"b.mc{lvl}_mthd AS mthd", f"b.mc{lvl}_mthd_id AS mthd_id"]
    tbls = [f"mc{lvl}_aeid AS a", f"mc{lvl}_methods AS b"]
    qstring = f"SELECT {', '.join(flds)} " \
              f"FROM {', '.join(tbls)} " \
              f"WHERE a.mc{lvl}_mthd_id = b.mc{lvl}_mthd_id " \
              f"AND aeid IN ({aeid});"
    return query_db(query=qstring)
