from query_db import tcpl_query


def tcpl_mthd_load(lvl, aeid, verbose=False):
    flds = ["aeid", f"b.mc{lvl}_mthd AS mthd", f"b.mc{lvl}_mthd_id AS mthd_id"]
    tbls = [f"mc{lvl}_aeid AS a", f"mc{lvl}_methods AS b"]
    qstring = f"SELECT {', '.join(flds)} " \
              f"FROM {', '.join(tbls)} " \
              f"WHERE a.mc{lvl}_mthd_id = b.mc{lvl}_mthd_id " \
              f"AND aeid IN ({', '.join(str(aeid))});"

    if verbose:
        print(f"qstring: {qstring}")

    return tcpl_query(query=qstring)
