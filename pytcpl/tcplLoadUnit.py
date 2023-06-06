from query_db import tcpl_query


def tcpl_load_unit(aeid):
    qformat = """
    SELECT
      aeid,
      normalized_data_type AS resp_unit
    FROM
      assay_component_endpoint
    WHERE
      aeid IN ({});
    """

    aeid_str = ",".join(str(id) for id in aeid)
    qstring = qformat.format(aeid_str)

    dat = tcpl_query(query=qstring, verbose=False)

    if dat.shape[0] == 0:
        print("Warning: The given aeid(s) do not have response units.")
        return dat

    len_miss = sum(aeid not in dat["aeid"].values for aeid in aeid)
    if len_miss > 0:
        print(f"Warning: {len_miss} of the given aeid(s) do not have response units.")

    return dat
