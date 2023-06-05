from query_db import get_sqlalchemy_engine
import time

def tcplAppend(dat, tbl):
    engine = get_sqlalchemy_engine()
    with engine.begin() as connection:
        start_time = time.time()
        num_rows_affected = dat.to_sql(name=tbl, con=connection, if_exists="append", index=False)
        print(f"Append to {tbl} >> {num_rows_affected} affected rows >> {str(time.time() - start_time)} seconds.")
    return num_rows_affected