from query_db import tcplQuery

def tcplLoadConcUnit(spid):
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

    dat = tcplQuery(query=qstring)

    if dat.shape[0] == 0:
        # print("The given spid(s) do not have concentration units.")
        return dat

    len_miss = sum(spid not in dat["spid"].values for spid in spid)
    if len_miss > 0:
        print(f"{len_miss} of the given spid(s) do not have concentration units.")

    return dat
