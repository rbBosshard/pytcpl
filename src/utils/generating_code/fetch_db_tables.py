import os
import mysql.connector
import pandas as pd
from src.utils.constants import INPUT_DIR_PATH
from src.utils.query_db import get_db_config


def export_tables_to_parquet():
    """
    Export data from specified MySQL tables to Parquet files.
    """
    user, password, host, port, database = get_db_config()
    connection = mysql.connector.connect(host=host, user=user, password=password, database=database)

    tables = ["assay_component", "assay_component_endpoint",
              "assay_component_endpoint_descriptions",
              "sample", "chemical",
              "mc4_aeid", "mc5_aeid",
              "mc4_methods", "mc5_methods"]

    for table in tables:
        df = pd.read_sql(f'SELECT * FROM {table}', connection)
        destination_path = os.path.join(INPUT_DIR_PATH, f"{table}.parquet.gzip")
        df.to_parquet(destination_path, compression='gzip')

    connection.close()


if __name__ == "__main__":
    export_tables_to_parquet()
