from query_db import tcplQuery

def tcplDelete(tbl, fld, val):
    qformat = f"DELETE FROM {tbl} WHERE"
    qformat += f" {' AND '.join([f'{fld} IN (%s)' for _ in val])}"
    qformat += ";"

    if not isinstance(val, list):
        val = [val]
    val = [','.join([f'"{x}"' for x in v]) for v in val]

    qstring = qformat % tuple(val)
    print(f'query: {qstring}')
    res = tcplQuery(qstring)
    print(f'res: {res}')

