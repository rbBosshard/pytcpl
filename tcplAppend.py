import pandas as pd
from query_db import get_sqlalchemy_engine
from query_db import get_sqlalchemy_engine 

def tcplAppend(dat, tbl):
    engine = get_sqlalchemy_engine()
    with engine.begin() as connection:
        num_rows_affected = dat.to_sql(name=tbl, con=connection, if_exists="append", index=False)
    return num_rows_affected