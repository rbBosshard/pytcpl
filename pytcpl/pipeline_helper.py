import os
import time

import numpy as np
import pandas as pd
import yaml

from mc4_mthds import mc4_mthds
from mc5_mthds import mc5_mthds
from symbols import symbols_dict
from tcpl_output import tcpl_output
from query_db import get_sqlalchemy_engine
from query_db import query_db
from tcpl_mthd_load import tcpl_mthd_load

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FOLDER_PATH = os.path.join(ROOT_DIR, 'config')
CONFIG_FILE_PATH = os.path.join(CONFIG_FOLDER_PATH, 'config.yaml')
AEIDS_LIST_PATH = os.path.join(ROOT_DIR, 'config', 'aeid_list.in')
DDL_PATH = os.path.join(ROOT_DIR, 'DDLs_new')
START_TIME = time.time()
DISPLAY_EMOJI = True

COLORS_DICT = {
    "WHITE": "\033[37m",
    "BLUE": "\033[34m",
    "GREEN": "\033[32m",
    "RED": "\033[31m",
    "ORANGE": "\033[33m",
    "VIOLET": "\033[35m",
    "RESET": "\033[0m",
}

# Custom format for tqdm progress bar (using blue color)
custom_format = f"{COLORS_DICT['WHITE']}{{desc}} {{percentage:3.0f}}%{{bar}} {{n_fmt}}/{{total_fmt}} {{elapsed}}<{{remaining}}{COLORS_DICT['RESET']}"


def text_to_blue(message):
    return f"{COLORS_DICT['BLUE']}{message}{COLORS_DICT['RESET']}"


def text_to_green(message):
    return f"{COLORS_DICT['GREEN']}{message}{COLORS_DICT['RESET']}" if DISPLAY_EMOJI else message


def status(symbol, replacement=""):
    return symbols_dict.get(symbol, "") if DISPLAY_EMOJI else replacement


def launch(config, confg_path):
    global DISPLAY_EMOJI
    aeid_list, aeid_list_path = read_aeids()
    DISPLAY_EMOJI = config['display_emoji']
    print(  f"{status('balloon')} Hi :)\n\n"
            f"{status('rocket')} Pytcpl launched!\n\n"
            f"{status('gear')} Configuration located in {confg_path}\n\n"
            f"{status('scroll')} Running pipeline for "
            f"{len(aeid_list)} assay endpoints (specified in 'config/aeid_list.in')\n")
    check_db(config)
    return aeid_list


def print_(msg):
    text = get_msg_with_elapsed_time(msg)
    print(text)


def get_msg_with_elapsed_time(msg, color_only_time=True):
    text = f"{get_formatted_time_elapsed(START_TIME, color_only_time)} {msg}"
    return text


def load_config():
    with open(CONFIG_FILE_PATH, 'r') as file:
        config = yaml.safe_load(file)
    return config, CONFIG_FOLDER_PATH


def prolog(new_aeid, config):
    # Update the specific key with the new value
    config['aeid'] = new_aeid

    # Write the updated YAML content back to the file
    with open(CONFIG_FILE_PATH, 'w') as file:
        yaml.dump(config, file)

    assay_component_endpoint_name = get_assay_info(config['aeid'])['assay_component_endpoint_name']

    print("\n" + f"#-" * 55 + "\n")
    assay_info = text_to_blue(f"{assay_component_endpoint_name} (aeid={config['aeid']})")
    print_(f"{status('seedling')} Start new assay endpoint: {assay_info}")


def check_db(config):
    if config['drop_new_tables']:
        new_table_names = config['new_table_names']
        tables = ", ".join(new_table_names)
        drop_stmt = f"DROP TABLE IF EXISTS {tables};"
        query_db(drop_stmt)  # Permanently removes tables from db!
        print(f"{status('broom')} Dropped all relevant tables")
    for ddl_file in os.scandir(DDL_PATH):
        with open(ddl_file, 'r') as f:
            ddl_query = f.read()
            query_db(ddl_query)
    print(f"{status('thumbs_up')} Verified the existence of required DB tables")


def get_formatted_time_elapsed(start_time, blue=True):
    seconds = time.time() - start_time
    minutes = int(seconds)
    formatted_seconds = "{:.2f}".format(seconds - minutes)[2:]
    elapsed_time_formatted = f"[0:{minutes:02}:{formatted_seconds}s]"
    if DISPLAY_EMOJI and blue:
        return text_to_blue(f"{elapsed_time_formatted}")
    else:
        return elapsed_time_formatted


def get_efficacy_cutoff(aeid, bmad):
    assay_cutoff_methods = tcpl_mthd_load(lvl=5, aeid=aeid)['mthd']
    cutoffs = [mc5_mthds(mthd, bmad) for mthd in assay_cutoff_methods]
    return max(cutoffs, default=0)


def track_fitted_params():
    tracked_models, tracked_parameters = read_log_file("fit_results_log.txt")
    parameters = {}
    for model, params in zip(tracked_models, tracked_parameters):
        if model not in parameters:
            parameters[model] = []
        else:
            parameters[model].append(params)
    try:
        with open("tracking_results.txt", "w") as file:
            for key, array in parameters.items():
                average = np.median(array, 0)
                file.write(f"{key} {average}\n")
    except Exception as e:
        print(f"{e}")


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


def store_cutoff(aeid, df):
    bmad = df["bmad"].iloc[0]
    cutoff = get_efficacy_cutoff(aeid=aeid, bmad=bmad)
    tcpl_append(pd.DataFrame({'aeid': [aeid], 'bmad': [bmad], 'cutoff': [cutoff]}), 'cutoffs')
    return cutoff


def get_efficacy_cutoff_and_append(aeid, df):
    get_bmad = tcpl_mthd_load(lvl=4, aeid=aeid)
    for mthd in list(get_bmad['mthd'].values):  # +['onesd.aeid.lowconc.twells']
        df = mc4_mthds(mthd)(df)
    bmad = df['bmad'].iloc[0]
    cutoff = get_efficacy_cutoff(aeid=aeid, bmad=bmad)
    tcpl_delete(aeid, 'cutoffs')
    tcpl_append(pd.DataFrame({'aeid': [aeid], 'bmad': [bmad], 'cutoff': [cutoff]}), 'cutoffs')
    return cutoff, df


def get_assay_info(aeid):
    qstring = f"SELECT * FROM assay_component_endpoint WHERE aeid = {aeid};"
    assay_component_endpoint = query_db(qstring)
    acid = assay_component_endpoint.iloc[0]["acid"]
    qstring = f"SELECT * FROM assay_component WHERE acid = {acid};"
    assay_component = query_db(qstring)
    assay_info_dict = pd.merge(assay_component_endpoint, assay_component, on='acid').iloc[0].to_dict()
    return assay_info_dict


def load_raw_data_from_db(aeid):
    print_(f"{status('hourglass_not_done')} Fetching raw data from DB..")
    select_cols = ['mc3.m0id', 'mc3.m1id', 'mc3.m2id', 'm3id', 'spid', 'aeid', 'logc', 'resp', 'cndx', 'wllt']
    qstring = f"SELECT {', '.join(select_cols)} " \
              f"FROM mc0, mc1, mc3 " \
              f"WHERE mc0.m0id = mc1.m0id AND mc1.m0id = mc3.m0id " \
              f"AND aeid = {aeid};"
    df = query_db(query=qstring)
    print_(f"{status('information')} Loaded {df.shape[0]} single datapoints")
    return df


def tcpl_append(dat, tbl):
    try:
        engine = get_sqlalchemy_engine()
        dat.to_sql(tbl, engine, if_exists='append', index=False)
    except Exception as err:
        print(err)


def tcpl_delete(aeid, tbl):
    qstring = f"DELETE FROM {tbl} WHERE aeid = {aeid};"
    query_db(qstring)


def read_aeids():
    with open(AEIDS_LIST_PATH, 'r') as file:
        ids_list = [line.strip() for line in file]
    return ids_list, AEIDS_LIST_PATH


def export_as_csv(config, dat):
    print_(f"{status('floppy_disk')} Exporting output data as csv")
    dat = dat[config['export_cols']]
    dat = tcpl_output(dat, config['aeid'])
    dat = dat.rename(columns={'dsstox_substance_id': 'dtxsid'})
    df_export = dat[['dtxsid', 'chit']]
    export_data(df_export, path=config['export_path'], folder='out', id=config['aeid'])
