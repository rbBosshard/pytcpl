import pandas as pd

from get_db_conn import get_db_conn

def tcplQuery(query):
    db_con = get_db_conn()
    try:
        df = pd.read_sql_query(query, db_con)
    except Exception as e:
        print(f"Error querying MySQL: {e}")
        return None
    db_con.close()
    return df

