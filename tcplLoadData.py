from query_db import tcplQuery
import yaml

mc4_name = "mc4_"
mc4_agg_name = "mc4_agg_"
mc4_param_name = "mc4_param_"
mc5_name = "mc5_"
mc5_param_name = "mc5_param_"


def prepField(fld, tbl, db):
    tbl_flds = [tcplListFlds(t, db) for t in tbl]
    pre = None
    for i in range(len(tbl)):
        if fld in tbl_flds[i]:
            pre = tbl[i]
                
    if pre is None:
        raise ValueError("Not all given fields available in query.")
        
    return pre + "." + fld


def tcplListFlds(tbl, db):
    qformat = """
    SELECT 
        `COLUMN_NAME` 
    FROM 
        `INFORMATION_SCHEMA`.`COLUMNS` 
    WHERE 
        `TABLE_SCHEMA` = '{db}' 
        AND 
        `TABLE_NAME` = '{tbl}';
    """   
    query = tcplQuery(qformat.format(db=db, tbl=tbl))["COLUMN_NAME"].tolist()
    return query


def tcplLoadData(lvl, fld=None, val=None, type="mc", add_fld=True):
    tbls = None
    
    if lvl == 0 and type == "mc":
        tbls = ["mc0"]
        cols = ["m0id","spid","acid","apid","rowi","coli","wllt","wllq","conc","rval","srcf"]
        col_str = ",".join(cols)
        qformat = f"SELECT {col_str} FROM mc0 "
        
    if lvl == 1 and type == "mc":
        tbls = ["mc0", "mc1"]
        qformat = (
            "SELECT mc1.m0id, m1id, spid, mc1.acid, apid, rowi, coli, wllt, wllq, "
            "conc, rval, cndx, repi, srcf "
            "FROM mc0, mc1 "
            "WHERE mc0.m0id = mc1.m0id "
        )
    
    if lvl == 2 and type == "mc":
        tbls = ["mc0", "mc1", "mc2"]
        qformat = (
            "SELECT mc2.m0id, mc2.m1id, m2id,spid, mc2.acid, apid, rowi, coli, wllt, conc, cval, cndx, repi "
            "FROM mc0, mc1, mc2 "
            "WHERE mc0.m0id = mc1.m0id AND mc1.m0id = mc2.m0id "
        )
    
    if lvl == 3 and type == "mc":
        tbls = ["mc0", "mc1", "mc3"]
        qformat = (
           "SELECT mc3.m0id, mc3.m1id, mc3.m2id, m3id, spid, aeid, logc, resp, cndx, wllt, apid, rowi, coli, repi "
            "FROM mc0, mc1, mc3 "
            "WHERE mc0.m0id = mc1.m0id AND mc1.m0id = mc3.m0id "
      )
    
    if lvl == "agg":
        tbls = ["mc3", f"{mc4_agg_name}", f"{mc4_name}"]
        qformat = (
            f"SELECT {mc4_agg_name}.aeid, {mc4_agg_name}.m4id, {mc4_agg_name}.m3id, {mc4_agg_name}.m2id, {mc4_agg_name}.m1id, {mc4_agg_name}.m0id, {mc4_name}.spid, logc, resp "
            f"FROM mc3, {mc4_agg_name}, {mc4_name} "
            f"WHERE mc3.m3id = {mc4_agg_name}.m3id "
            f"AND {mc4_name}.m4id = {mc4_agg_name}.m4id "
        )
    
    if lvl == 4 and type == "mc":
        tbls = [f"{mc4_name}", f"{mc4_param_name}"]
        qformat = (
            f"SELECT {mc4_name}.m4id,{mc4_name}.aeid,spid,bmad,resp_max,resp_min,max_mean,max_mean_conc,max_med,max_med_conc,logc_max,logc_min,nconc,npts,nrep,nmed_gtbl,model,model_param,model_val "
            f"FROM {mc4_param_name}, {mc4_name} "
            f"WHERE {mc4_name}.m4id = {mc4_param_name}.m4id "
        )

    if lvl == 5 and type == "mc":
        tbls = [f"{mc4_name}", f"{mc5_name}", f"{mc5_param_name}"]
        qformat = (
            f"SELECT {mc5_name}.m5id,{mc5_name}.m4id,{mc5_name}.aeid,spid,bmad,resp_max,resp_min,max_mean,max_mean_conc,max_med,max_med_conc,logc_max,logc_min,nconc,npts,nrep,nmed_gtbl,hitc,modl,fitc,coff,hit_param,hit_val "
            f"FROM {mc4_name}, {mc5_name}, {mc5_param_name} "
            f"WHERE {mc4_name}.m4id = {mc5_name}.m4id "
            f"AND {mc5_name}.m5id = {mc5_param_name}.m5id "
       )

    if tbls is None:
        raise ValueError("Invalid 'lvl' and 'type' combination.")
    
    if fld is not None:
        if val is None:
            raise ValueError("'val' cannot be None. Please provide a valid value for the specified field.")

        with open("config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            db = config['DATABASE']['DB']

        fld = prepField(fld = fld, tbl = tbls, db = db)
        
        if add_fld:
            wtest = False
        wtest = lvl in [0] or (lvl == 2 and type == "sc")
        # if not check_tcpl_db_schema() and lvl == 4:
        #     wtest = True
        
        qformat = qformat + ("WHERE" if wtest else "AND")
        
        if isinstance(fld, str):
            fld = [fld]
        qformat += "  " + " AND ".join([f"{fld[i]} IN (%s)" for i in range(len(fld))])
        qformat += ";"
        # print(f"qformat: {qformat}")
        if not isinstance(val, list):
            val = [val]


        val = ','.join([str(v) for v in val])
        qstring = qformat % val
        # print(f"qstring: {qstring}")

    else:
        qstring = qformat
    
    dat = tcplQuery(query = qstring)
    
    # pivot table so 1 id per return and only return added fields
    # if add_fld and check_tcpl_db_schema():
    #     if lvl == 4:
    #         dat = pd.pivot_wider(dat, names_from = [model,model_param], values_from = model_val)
    #     if lvl == 5:
    #         dat = pd.pivot_wider(dat, names_from = hit_param, values_from = hit_val)
    
    return dat