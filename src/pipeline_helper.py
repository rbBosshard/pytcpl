import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from constants import COLORS_DICT, CONFIG_DIR_PATH, CONFIG_PATH, AEIDS_LIST_PATH, DDL_PATH, \
    CSV_DIR_PATH, LOG_DIR_PATH, START_TIME
from constants import symbols_dict
from mthds import mc4_mthds, mc5_mthds, tcpl_mthd_load
from query_db import get_sqlalchemy_engine
from query_db import query_db
from tcpl_output import tcpl_output
import matplotlib.pyplot as plt
from fit_models import get_params

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
    DISPLAY_EMOJI = 0 if '--unicode' in sys.argv else config['enable_fancy_logging']

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


def bye():
    print(f"\n{status('confetti_ball')} Pipeline completed!")
    print(f"{status('waving_hand')} Goodbye")


def check_db(CONFIG):
    if CONFIG['enable_dropping_new_tables']:
        new_table_names = CONFIG['new_db_tables']
        tables = ", ".join(new_table_names)
        drop_stmt = f"DROP TABLE IF EXISTS {tables};"
        query_db(drop_stmt)  # Permanently removes tables from db!
        print(f"{status('broom')} Dropped all relevant tables")
    for ddl_file in os.scandir(DDL_PATH):
        with open(ddl_file, 'r') as f:
            ddl_query = f.read()
            query_db(ddl_query)
    print(f"{status('thumbs_up')} Verified the existence of required DB tables")


def track_fitted_params(fit_params):
    parameters = {}
    for res in fit_params:
        for model, params in res.items():
            ps = list(params['pars'].values())
            parameters.setdefault(model, []).append(ps)

    def plot_histograms(params, key):
        plt.figure(figsize=(10, 6))
        plt.suptitle(f'Fit model parameters histograms: {key}')
        param_names = get_params(key)
        num_params = len(param_names)
        for i in range(num_params):
            plt.subplot(1, num_params, i + 1)
            plt.hist(params[:, i])
            plt.title(f"{param_names[i]}")
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_DIR_PATH, "curve_fit_parameter_tracks", f"{key}.png"))

    with open(os.path.join(LOG_DIR_PATH, "curve_fit_parameter_tracks", f"stats.out"), "w") as file:
        # Plot histograms for each key in the dictionary
        for key, param_list in parameters.items():
            param_array = np.array(param_list)
            plot_histograms(param_array, key)
            median = np.median(param_array, 0)
            min = np.min(param_array, 0)
            max = np.max(param_array, 0)
            file.write(f"{key}:\n >> median {median}\n >> min {min}\n >> max {max}\n\n")


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


def get_cutoffs(df):
    print_(f"{status('thermometer')} Computing efficacy cutoff")
    values = {}
    other_mthds = ['bmed.aeid.lowconc.twells', 'onesd.aeid.lowconc.twells']
    for mthd in tcpl_mthd_load(lvl=4, aeid=CONFIG['aeid']) + other_mthds:
        value = mc4_mthds(mthd, df)
        values.update({mthd: value})

    bmad = values['bmad.aeid.lowconc.twells']
    bmed = values['bmed.aeid.lowconc.twells']
    onesd = values['onesd.aeid.lowconc.twells']

    cutoffs = [mc5_mthds(mthd, bmad) for mthd in tcpl_mthd_load(lvl=5, aeid=CONFIG['aeid'])]
    cutoff = max(cutoffs, default=0)

    tcpl_delete(CONFIG['aeid'], 'cutoffs')
    tcpl_append(pd.DataFrame({'aeid': [CONFIG['aeid']],
                              'bmad': [bmad],
                              'bmed': [bmed],
                              'onesd': [onesd],
                              'cutoff': [cutoff]}), 'cutoffs')
    return cutoff


def get_assay_info(aeid):
    qstring = f"SELECT * FROM assay_component_endpoint WHERE aeid = {aeid};"
    assay_component_endpoint = query_db(qstring)
    acid = assay_component_endpoint.iloc[0]["acid"]
    qstring = f"SELECT * FROM assay_component WHERE acid = {acid};"
    assay_component = query_db(qstring)
    assay_info_dict = pd.merge(assay_component_endpoint, assay_component, on='acid').iloc[0].to_dict()
    return assay_info_dict


def load_raw_data():
    suffix = ".csv.gzip" if CONFIG['data_file_format'] == 'csv' else ".parquet.gzip"
    csv_file_path = os.path.join(CSV_DIR_PATH, f"{CONFIG['aeid']}_in{suffix}")
    load_from_disk = os.path.exists(csv_file_path)
    data_source = "disk" if load_from_disk else "DB"

    print_(f"{status('hourglass_not_done')} Fetching raw data from {data_source}..")

    if load_from_disk:
        df = pd.read_parquet(csv_file_path) if CONFIG['data_file_format'] == 'csv' else pd.read_parquet(csv_file_path)
    else:
        select_cols = ['spid', 'aeid', 'logc', 'resp', 'cndx', 'wllt']
        table_mapping = {'mc0.m0id': 'mc1.m0id', 'mc1.m0id': 'mc3.m0id'}
        join_string = ' AND '.join([f"{key} = {value}" for key, value in table_mapping.items()])
        qstring = f"SELECT {', '.join(select_cols)} " \
                  f"FROM mc0, mc1, mc3 " \
                  f"WHERE {join_string} AND aeid = {CONFIG['aeid']};"
        df = query_db(query=qstring)
        if CONFIG['data_file_format'] == 'csv':
            df.to_csv(csv_file_path, compression='gzip')
        else:
            df.to_parquet(csv_file_path, compression='gzip')

    print_(f"{status('information')} Loaded {df.shape[0]} single datapoints")

    cutoff = get_cutoffs(df)

    return df, cutoff


def tcpl_append(dat, tbl):
    try:
        engine = get_sqlalchemy_engine()
        dat.to_sql(tbl, engine, if_exists='append', index=False)
    except Exception as err:
        print(err)


def tcpl_delete(aeid, tbl):
    qstring = f"DELETE FROM {tbl} WHERE aeid = {aeid};"
    query_db(qstring)


def store_output_in_db(df):
    mb_value = f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
    print_(f"{status('computer_disk')} Writing output data to DB (~{mb_value})..")
    for col in ['conc', 'response', 'fit_params']:
        df.loc[:, col] = df[col].apply(json.dumps)
    tcpl_delete(CONFIG['aeid'], "output")
    tcpl_append(df[CONFIG['z_output_columns']], "output")


def export(df):
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
    elapsed_time_formatted = f"[{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{hundredths:02}]"

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
