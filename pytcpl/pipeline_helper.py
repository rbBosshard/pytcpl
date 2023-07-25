import os
import time
import numpy as np

import yaml

from query_db import tcpl_query
from tcpl_mthd_load import tcpl_mthd_load
from mc5_mthds import mc5_mthds

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config', 'config.yaml')
DDL_PATH = os.path.join(ROOT_DIR, 'DDLs_slim')

def load_config():
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    return config


def starting(pipeline_step):
    print(f"Starting: {pipeline_step}..")
    return time.time()


def elapsed(start_time):
    return f"Execution time in seconds: {str(round(time.time() - start_time, 2))}"


def get_cutoff(aeid, bmad):
    assay_cutoff_methods = tcpl_mthd_load(lvl=5, aeid=aeid)["mthd"]
    cutoffs = [mc5_mthds(mthd, bmad) for mthd in assay_cutoff_methods]
    return max(cutoffs, default=0)

def get_mc5_data(aeid):
    mc4_name = "mc4_"
    mc4_param_name = "mc4_param_"
    query = f"SELECT {mc4_name}.m4id," \
            f"{mc4_name}.aeid," \
            f"{mc4_param_name}.model," \
            f"{mc4_param_name}.model_param," \
            f"{mc4_param_name}.model_val " \
            f"FROM {mc4_name} " \
            f"JOIN {mc4_param_name} " \
            f"ON {mc4_name}.m4id = {mc4_param_name}.m4id " \
            f"WHERE {mc4_name}.aeid = {aeid};"

    dat = tcpl_query(query)
    return dat


def track_fitted_params():
    tracked_models, tracked_parameters = read_log_file("fit_results_log.txt")
    parameters = {}
    for model, params in zip(tracked_models, tracked_parameters):
        if model not in parameters:
            parameters[model] = []
        else:
            parameters[model].append(params)
    with open("tracking_results.txt", "w") as file:
        for key, array in parameters.items():
            average = np.median(array, 0)
            file.write(f"{key} {average}\n")

def get_my_data(aeid):
    mc4_name = "mc4_"
    mc4_param_name = "mc4_param_"
    query = f"SELECT {mc4_name}.m4id," \
            f"{mc4_name}.aeid," \
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
    start_time = starting(f"Drop all tables")
    tables = ", ".join(table_names_list)
    drop_stmt = f"DROP TABLE {tables};"
    tcpl_query(drop_stmt)  # Permanently removes tables tables from db!
    print(f"Done >> {elapsed(start_time)}\n")


def ensure_all_new_db_tables_exist():
    start_time = starting(f"Ensure that all tables exist")
    for ddl_file in os.scandir(DDL_PATH):
        with open(ddl_file, 'r') as f:
            ddl_query = f.read()
            tcpl_query(ddl_query)
    print(f"Done >> {elapsed(start_time)}\n")


def export_data(dat, path, folder, id):
    full_folder_path = os.path.join(ROOT_DIR, path, folder)
    is_exist = os.path.exists(full_folder_path)
    if not is_exist:
        os.makedirs(full_folder_path)
    dat.to_csv(f"{full_folder_path}/{id}.csv", index=False)


def read_log_file(log_file):
    models = []
    parameters = []
    with open(log_file, "r") as file:
        for line in file:
            model, params = line.strip().split(": ")
            param_list = []
            for x in params.split(", "):
                if x != 'None':
                    param_list.append(float(x))
                else:
                    continue
            if param_list:
                models.append(model)
                parameters.append(param_list)
    return models, parameters
