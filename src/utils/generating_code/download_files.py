import pandas as pd
import mysql.connector
import os

from src.utils.constants import INPUT_DIR_PATH
from src.utils.constants import INPUT_DIR_PATH

# MySQL connection parameters
host = 'localhost'
user = 'root'
password = 'root'
database = 'invitrodb_v3o5'
table_name = 'assay_component_endpoint_descriptions'
# assay_component, assay_component_endpoint, sample, chemical, assay_component_endpoint_descriptions

# Establish a connection
connection = mysql.connector.connect(host=host, user=user, password=password, database=database)

# SQL query
query = f'SELECT * FROM {table_name}'

# Download data into a DataFrame
df = pd.read_sql(query, connection)
destination_path = os.path.join(INPUT_DIR_PATH, f"{table_name}.parquet.gzip")
df.to_parquet(destination_path, compression='gzip')
# Close the connection
connection.close()
