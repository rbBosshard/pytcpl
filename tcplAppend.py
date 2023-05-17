import pandas as pd
from get_db_conn import get_db_conn

def tcplAppend(dat, tbl):
    db_con = get_db_conn()
    num_rows_affected = dat.to_sql(name=tbl, con=db_con, if_exists="append", index=False)
    db_con.close()
    return num_rows_affected