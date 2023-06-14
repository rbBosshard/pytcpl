import os
import time

import yaml

from query_db import tcpl_query

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, '../config/config.yaml')

def load_config():
    with open(os.path.join(CONFIG_PATH)) as file:
        config = yaml.safe_load(file)
    return config


def starting(pipeline_step):
    print(f"Starting: {pipeline_step}..")
    return time.time()


def elapsed(start_time):
    return f"Execution time in seconds: {str(round(time.time() - start_time, 2))}"


def get_mc5_data(aeid):
    mc4_name = "mc4_"
    mc4_param_name = "mc4_param_"
    query = f"SELECT {mc4_name}.m4id," \
            f"{mc4_name}.aeid," \
            f"{mc4_name}.logc_max," \
            f"{mc4_name}.logc_min," \
            f"{mc4_param_name}.model," \
            f"{mc4_param_name}.model_param," \
            f"{mc4_param_name}.model_val " \
            f"FROM {mc4_name} " \
            f"JOIN {mc4_param_name} " \
            f"ON {mc4_name}.m4id = {mc4_param_name}.m4id " \
            f"WHERE {mc4_name}.aeid = {aeid};"

    dat = tcpl_query(query)
    return dat


def drop_tables(table_names_list):
    # Permanently removes tables tables from db!
    start_time = starting(f"Drop all tables")

    tables = ", ".join(table_names_list)
    drop_stmt = f"DROP TABLE {tables};"
    tcpl_query(drop_stmt)

    print(f"Done >> {elapsed(start_time)}")


def ensure_all_new_db_tables_exist():
    start_time = starting(f"Ensure that all tables exist")

    ddl_directory = 'DDLs'
    for ddl_file in os.scandir(ddl_directory):
        with open(ddl_file, 'r') as f:
            ddl_query = f.read()
            tcpl_query(ddl_query)

    print(f"Done >> {elapsed(start_time)}")


def export_data(dat, path, folder, id):
    full_folder_path = path + folder
    isExist = os.path.exists(full_folder_path)
    if not isExist:
        os.makedirs(full_folder_path)
    dat.to_csv(f"{full_folder_path}/{id}.csv")
