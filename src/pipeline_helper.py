import json
import os
import sys
from datetime import datetime
import time

import numpy as np
import pandas as pd
import yaml

from constants import symbols_dict
from mthds import mc4_mthds, mc5_mthds, tcpl_mthd_load
from query_db import get_sqlalchemy_engine
from query_db import query_db
from constants import COLORS_DICT, CONFIG_DIR_PATH, CONFIG_PATH, AEIDS_LIST_PATH, DDL_PATH, \
    CSV_DIR_PATH, LOG_DIR_PATH, START_TIME
from tcpl_output import tcpl_output

CONFIG = {}
DISPLAY_EMOJI = 1


def status(symbol, replacement=""):
    return symbols_dict.get(symbol, "") if DISPLAY_EMOJI else replacement


def read_aeids():
    with open(AEIDS_LIST_PATH, 'r') as file:
        ids_list = [line.strip() for line in file]
    return ids_list


def launch(config, config_path):
    global DISPLAY_EMOJI
    aeid_list = read_aeids()
    # disable verbose output if --unicode passed as runtime argument
    DISPLAY_EMOJI = 0 if '--unicode' in sys.argv else config['apply_fancy_logging']

    print(f"{status('balloon')} Hi :)\n"
          f"{status('rocket')} Pytcpl launched!\n"
          f"{status('gear')} Configuration located in {config_path}\n"
          f"{status('scroll')} Running pipeline for "
          f"{len(aeid_list)} assay endpoints (specified in 'config/aeid_list.in')")
    check_db(config)
    return aeid_list


def load_config():
    with open(CONFIG_PATH, 'r') as file:
        CONFIG = yaml.safe_load(file)
    return CONFIG, CONFIG_DIR_PATH


def prolog(config, new_aeid):
    global CONFIG
    CONFIG = config
    
    # print  boundary
    print("\n" + f"#-" * 55 + "\n")

    # Update the specific key with the new value
    CONFIG['aeid'] = new_aeid

    # Write the updated YAML content back to the file
    with open(CONFIG_PATH, 'w') as file:
        yaml.dump(CONFIG, file)

    assay_component_endpoint_name = get_assay_info(CONFIG['aeid'])['assay_component_endpoint_name']
    assay_info = text_to_blue(f"{assay_component_endpoint_name} (aeid={CONFIG['aeid']})")
    print_(f"{status('seedling')} Start processing new assay endpoint: {assay_info}")


def epilog():
    print_(f"{status('carrot')} Assay endpoint processing completed")


def goodbye():
    print(f"\n{status('clinking_beer_mugs')} Pipeline completed")
    print(f"{status('waving_hand')} Goodbye!")


def check_db(CONFIG):
    if CONFIG['apply_dropping_new_tables']:
        new_table_names = CONFIG['new_table_names']
        tables = ", ".join(new_table_names)
        drop_stmt = f"DROP TABLE IF EXISTS {tables};"
        query_db(drop_stmt)  # Permanently removes tables from db!
        print(f"{status('broom')} Dropped all relevant tables")
    for ddl_file in os.scandir(DDL_PATH):
        with open(ddl_file, 'r') as f:
            ddl_query = f.read()
            query_db(ddl_query)
    print(f"{status('thumbs_up')} Verified the existence of required DB tables")


def get_efficacy_cutoff(aeid, bmad):
    assay_cutoff_methods = tcpl_mthd_load(lvl=5, aeid=aeid)['mthd']
    cutoffs = [mc5_mthds(mthd, bmad) for mthd in assay_cutoff_methods]
    return max(cutoffs, default=0)


def track_fitted_params():
    tracked_models, tracked_params = read_log_file(os.path.join(LOG_DIR_PATH, "params_tracked.out"))
    params = {}
    for m, ps in zip(tracked_models, tracked_params):
        if m not in params:
            params[m] = []
        else:
            params[m].append(ps)
    try:
        with open(os.path.join(LOG_DIR_PATH, "params_tracked_median.out"), "w") as file:
            for key, array in params.items():
                average = np.median(array, 0)
                file.write(f"{key} {average}\n")
    except Exception as e:
        print(f"{e}")


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


def get_efficacy_cutoff_and_append(df):
    get_bmad = tcpl_mthd_load(lvl=4, aeid=CONFIG['aeid'])
    for mthd in list(get_bmad['mthd'].values):  # +['onesd.aeid.lowconc.twells']
        df = mc4_mthds(mthd)(df)
    bmad = df['bmad'].iloc[0]
    cutoff = get_efficacy_cutoff(aeid=CONFIG['aeid'], bmad=bmad)
    tcpl_delete(CONFIG['aeid'], 'cutoffs')
    tcpl_append(pd.DataFrame({'aeid': [CONFIG['aeid']], 'bmad': [bmad], 'cutoff': [cutoff]}), 'cutoffs')
    return cutoff, df


def get_assay_info(aeid):
    qstring = f"SELECT * FROM assay_component_endpoint WHERE aeid = {aeid};"
    assay_component_endpoint = query_db(qstring)
    acid = assay_component_endpoint.iloc[0]["acid"]
    qstring = f"SELECT * FROM assay_component WHERE acid = {acid};"
    assay_component = query_db(qstring)
    assay_info_dict = pd.merge(assay_component_endpoint, assay_component, on='acid').iloc[0].to_dict()
    return assay_info_dict


def load_raw_data():
    csv_file_path = os.path.join(CSV_DIR_PATH, f"{CONFIG['aeid']}_in.csv")
    load_from_csv = os.path.exists(csv_file_path)
    data_source = "csv file" if load_from_csv else "DB"
    
    print_(f"{status('hourglass_not_done')} Fetching raw data from {data_source}..")
    
    if load_from_csv:
        df = pd.read_csv(csv_file_path)
    else:   
        select_cols = ['mc3.m0id', 'mc3.m1id', 'mc3.m2id', 'm3id', 'spid', 'aeid', 'logc', 'resp', 'cndx', 'wllt']
        qstring = f"SELECT {', '.join(select_cols)} " \
                  f"FROM mc0, mc1, mc3 " \
                  f"WHERE mc0.m0id = mc1.m0id AND mc1.m0id = mc3.m0id " \
                  f"AND aeid = {CONFIG['aeid']};"
        df = query_db(query=qstring)
        df.to_csv(csv_file_path)

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


def write_output_data_to_db(df):
    mb_value = f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
    print_(f"{status('computer_disk')} Writing output data to DB (~{mb_value})..")
    for col in ['concentration_unlogged', 'response', 'fitparams']:
        df.loc[:, col] = df[col].apply(json.dumps)
    tcpl_delete(CONFIG['aeid'], "output")
    tcpl_append(df[CONFIG['z_output_columns']], "output")


def export_as_csv(df):
    print_(f"{status('floppy_disk')} Exporting output data as csv")
    df = df[CONFIG['z_export_cols']]
    df = tcpl_output(df, CONFIG['aeid'])
    df = df.rename(columns={'dsstox_substance_id': 'dtxsid'})
    df = df[['dtxsid', 'chit']]
    df.to_csv(f"{CSV_DIR_PATH}/{CONFIG['aeid']}.csv", index=False)


def text_to_blue(message):
    return f"{COLORS_DICT['BLUE']}{message}{COLORS_DICT['RESET']}" if DISPLAY_EMOJI else message


def get_formatted_time_elapsed(start_time, blue=True):
    delta = datetime.now() - start_time
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    hundredths = int((delta.microseconds / 10000) % 100)
    elapsed_time_formatted =  f"[{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{hundredths:02}]"

    if DISPLAY_EMOJI and blue:
        return text_to_blue(f"{elapsed_time_formatted}")
    else:
        return elapsed_time_formatted


def get_msg_with_elapsed_time(msg, color_only_time=True):
    text = f"{get_formatted_time_elapsed(START_TIME, color_only_time)} {msg}"
    return text


def print_(msg):
    text = get_msg_with_elapsed_time(msg)
    print(text)
