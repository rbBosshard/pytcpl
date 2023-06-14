from query_db import tcpl_query

mc4_name = "mc4_"
mc4_agg_name = "mc4_agg_"
mc4_param_name = "mc4_param_"
mc5_name = "mc5_"
mc5_param_name = "mc5_param_"


def prep_field(fld, tbls):
    # scope with the fully qualified table.field name name to avoid ambiguity
    for table in tbls:
        if fld in tcpl_list_flds(table):
            return f"{table}.{fld}"

def tcpl_list_flds(tbl):
    qformat = """
    SELECT 
        `COLUMN_NAME` 
    FROM 
        `INFORMATION_SCHEMA`.`COLUMNS` 
    WHERE 
        `TABLE_NAME` = '{tbl}';
    """
    query = tcpl_query(qformat.format(tbl=tbl))["COLUMN_NAME"].tolist()
    return query


def tcpl_load_data(lvl, fld, ids, verbose=False):
    if lvl == 3:
        tbls = ["mc0", "mc1", "mc3"]
        qformat = (
            "SELECT mc3.m0id, mc3.m1id, mc3.m2id, m3id, spid, aeid, logc, resp, cndx, wllt, apid, rowi, coli, repi "
            "FROM mc0, mc1, mc3 "
            "WHERE mc0.m0id = mc1.m0id AND mc1.m0id = mc3.m0id "
        )

    elif lvl == "agg":
        tbls = ["mc3", f"{mc4_agg_name}", f"{mc4_name}"]
        qformat = (
            f"SELECT {mc4_agg_name}.aeid,"
            f" {mc4_agg_name}.m4id,"
            f" {mc4_agg_name}.m3id,"
            f" {mc4_agg_name}.m2id,"
            f" {mc4_agg_name}.m1id,"
            f" {mc4_agg_name}.m0id,"
            f" {mc4_name}.spid,"
            f" logc,"
            f" resp "
            f"FROM mc3, {mc4_agg_name}, {mc4_name} "
            f"WHERE mc3.m3id = {mc4_agg_name}.m3id "
            f"AND {mc4_name}.m4id = {mc4_agg_name}.m4id "
        )

    elif lvl == 4:
        tbls = [f"{mc4_name}", f"{mc4_param_name}"]
        qformat = (
            f"SELECT {mc4_name}.m4id,{mc4_name}.aeid,"
            f"spid,bmad,resp_max,resp_min,max_mean,"
            f"max_mean_conc,max_med,max_med_conc,"
            f"logc_max,logc_min,nconc,npts,nrep,"
            f"nmed_gtbl,model,model_param,model_val "
            f"FROM {mc4_param_name}, {mc4_name} "
            f"WHERE {mc4_name}.m4id = {mc4_param_name}.m4id "
        )

    elif lvl == 5:
        tbls = [f"{mc4_name}", f"{mc5_name}", f"{mc5_param_name}"]
        qformat = (
            f"SELECT {mc5_name}.m5id,{mc5_name}.m4id,{mc5_name}.aeid,"
            f"spid,bmad,resp_max,resp_min,max_mean,max_mean_conc,max_med,"
            f"max_med_conc,logc_max,logc_min,nconc,npts,nrep,nmed_gtbl,"
            f"hitc,modl,fitc,coff,hit_param,hit_val "
            f"FROM {mc4_name}, {mc5_name}, {mc5_param_name} "
            f"WHERE {mc4_name}.m4id = {mc5_name}.m4id "
            f"AND {mc5_name}.m5id = {mc5_param_name}.m5id "
        )
    else:
        raise ValueError("lvl not supported")

    fld = prep_field(fld=fld, tbls=tbls)


    qformat = qformat + "AND "

    if isinstance(fld, str):
        fld = [fld]
    qformat += " AND ".join([f"{fld[i]} IN (%s)" for i in range(len(fld))])
    qformat += ";"

    if not isinstance(ids, list):
        ids = [ids]

    ids = ','.join([str(v) for v in ids])
    qstring = qformat % ids

    if verbose:
        print(f"qstring: {qstring}")

    dat = tcpl_query(query=qstring)

    return dat
