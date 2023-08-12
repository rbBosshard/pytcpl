import json
import os
import sys
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .constants import COLORS_DICT, CONFIG_DIR_PATH, CONFIG_PATH, AEIDS_LIST_PATH, DDL_PATH, \
    EXPORT_DIR_PATH, LOG_DIR_PATH, START_TIME, ERROR_PATH, RAW_DIR_PATH, OUTPUT_DIR_PATH, OUT_DIR_PATH, INPUT_DIR_PATH, \
    CUTOFF_DIR_PATH
from .constants import symbols_dict
from .get_model import get_model
from .mthds import mc4_mthds, mc5_mthds, tcpl_mthd_load
from .query_db import get_sqlalchemy_engine, query_db
from .query_db import query_db

CONFIG = {}
DISPLAY_EMOJI = 1
SUFFIX = ''


def status(symbol, replacement=""):
    return symbols_dict.get(symbol, "") if DISPLAY_EMOJI else replacement


def read_aeids():
    with open(AEIDS_LIST_PATH, 'r') as file:
        ids_list = [line.strip() for line in file]
    return ids_list


def launch(config, config_path):
    with open(ERROR_PATH, "w") as f:
        print("Failed assay endpoints:\n", file=f)

    global DISPLAY_EMOJI, SUFFIX
    # disable verbose output if --unicode passed as runtime argument
    DISPLAY_EMOJI = 0 if '--unicode' in sys.argv else config['enable_fancy_logging']

    aeid_list = read_aeids()
    print(f"{status('rocket')} Pytcpl launched!")
    print(f"{status('gear')} Configuration located in {config_path}")
    print(f"{status('scroll')} Running pipeline for {len(aeid_list)} assay endpoints (spec in 'config/aeid_list.in')")

    check_db(config)
    SUFFIX = f".{config['data_file_format']}.gzip"
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
    CONFIG['aeid'] = int(new_aeid)

    # Write the updated YAML content back to the file
    with open(CONFIG_PATH, 'w') as file:
        yaml.dump(CONFIG, file)

    assay_component_endpoint_name = get_assay_info(CONFIG['aeid'], CONFIG['enable_output_to_db'])['assay_component_endpoint_name']
    assay_info = text_to_blue(f"{assay_component_endpoint_name} (aeid={CONFIG['aeid']})")
    print_(f"{status('seedling')} Start processing new assay endpoint: {assay_info}")


def epilog():
    print_(f"{status('carrot')} Assay endpoint processing completed")


def bye():
    print(f"\n{status('confetti_ball')} Pipeline completed!")
    print(f"{status('waving_hand')} Goodbye")


def check_db(CONFIG):
    if CONFIG['enable_output_to_db']:
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
    for result in fit_params:
        for model, params in result.items():
            parameters.setdefault(model, []).append(list(params['pars'].values()))

    def plot_histograms(params, key):
        plt.figure(figsize=(10, 6))
        plt.suptitle(f'Fit model parameters histograms: {key}')
        param_names = get_model(key)('params')
        num_params = len(param_names)
        for i, param_name in enumerate(param_names):
            plt.subplot(1, num_params, i + 1)
            plt.hist(params[:, i], bins=50)
            plt.title(f"{param_name}")
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_DIR_PATH, "curve_fit_parameter_tracks", f"{key}.png"))
        plt.close()

    matplotlib.use('Agg')

    with open(os.path.join(LOG_DIR_PATH, "curve_fit_parameter_tracks", f"stats.out"), "w") as file:
        for key, param_list in parameters.items():
            param_array = np.array(param_list)
            plot_histograms(param_array, key)
            median, minimum, maximum = np.median(param_array, 0), np.min(param_array, 0), np.max(param_array, 0)
            file.write(f"{key}:\n >> median {median}\n >> min {minimum}\n >> max {maximum}\n\n")


def get_cutoffs(df):
    print_(f"{status('thermometer')} Computing efficacy cutoff")
    aeid = CONFIG['aeid']

    path = os.path.join(CUTOFF_DIR_PATH, f"{aeid}{SUFFIX}")
    if os.path.exists(path):
        out = pd.read_parquet(path)
        cutoff = out.iloc[0]['cutoff']
    else:
        values = {}
        other_mthds = ['bmed.aeid.lowconc.twells', 'onesd.aeid.lowconc.twells']
        for mthd in tcpl_mthd_load(lvl=4, aeid=aeid) + other_mthds:
            values.update({mthd: mc4_mthds(mthd, df)})

        bmad = values['bmad.aeid.lowconc.twells'] if 'bmad.aeid.lowconc.twells' in values else values['bmad.aeid.lowconc.nwells']
        bmed = values['bmed.aeid.lowconc.twells']
        sd = values['onesd.aeid.lowconc.twells']

        cutoffs = [mc5_mthds(mthd, bmad) for mthd in tcpl_mthd_load(lvl=5, aeid=aeid)]
        cutoffs = list(filter(lambda item: item is not None, cutoffs))
        cutoff = max(cutoffs, default=0)
        out = pd.DataFrame({'aeid': [aeid], 'bmad': [bmad], 'bmed': [bmed], 'onesd': [sd], 'cutoff': [cutoff]})

    if CONFIG['enable_output_to_db']:
        db_delete('cutoff')
        db_append(out, 'cutoff')
    return cutoff


def get_assay_info(aeid, from_db):
    tbl_endpoint = 'assay_component_endpoint'
    tbl_component = 'assay_component'

    if from_db:
        endpoint_query = f"SELECT * FROM {tbl_endpoint} WHERE aeid = {aeid};"
        acid = query_db(endpoint_query).iloc[0]['acid']
        component_query = f"SELECT * FROM {tbl_component} WHERE acid = {acid};"
        assay_info_dict = pd.merge(query_db(endpoint_query), query_db(component_query), on='acid').iloc[0].to_dict()
    else:
        endpoint_df = pd.read_parquet(os.path.join(INPUT_DIR_PATH, f"{tbl_endpoint}{SUFFIX}"))
        component_df = pd.read_parquet(os.path.join(INPUT_DIR_PATH, f"{tbl_component}{SUFFIX}"))
        assay_info_dict = pd.merge(endpoint_df, component_df, on='acid').iloc[0].to_dict()

    return assay_info_dict


def load_raw_data():
    csv_file_path = os.path.join(RAW_DIR_PATH, f"{CONFIG['aeid']}{SUFFIX}")
    load_from_disk = os.path.exists(csv_file_path)
    data_source = "disk" if load_from_disk else "DB"

    print_(f"{status('hourglass_not_done')} Fetching raw data from {data_source}..")

    if load_from_disk:
        df = pd.read_parquet(csv_file_path)
    else:
        select_cols = ['spid', 'aeid', 'logc', 'resp', 'cndx', 'wllt']
        table_mapping = {'mc0.m0id': 'mc1.m0id', 'mc1.m0id': 'mc3.m0id'}
        join_string = ' AND '.join([f"{key} = {value}" for key, value in table_mapping.items()])
        qstring = f"SELECT {', '.join(select_cols)} " \
                  f"FROM mc0, mc1, mc3 " \
                  f"WHERE {join_string} AND aeid = {CONFIG['aeid']};"
        df = query_db(query=qstring)
        df.to_parquet(csv_file_path, compression='gzip')

    print_(f"{status('information')} Loaded {df.shape[0]} single datapoints")
    cutoff = get_cutoffs(df)

    return df, cutoff


def db_append(dat, tbl):
    if CONFIG['enable_output_to_db']:
        try:
            engine = get_sqlalchemy_engine()
            dat.to_sql(tbl, engine, if_exists='append', index=False)
        except Exception as err:
            print(err)
    else:
        file_path = os.path.join(EXPORT_DIR_PATH, tbl, f"{CONFIG['aeid']}{SUFFIX}")
        dat.to_parquet(file_path, compression='gzip')


def db_delete(tbl):
    if CONFIG['enable_output_to_db']:
        query_db(f"DELETE FROM {tbl} WHERE aeid = {CONFIG['aeid']};")
    else:
        file_path = os.path.join(EXPORT_DIR_PATH, tbl, f"{CONFIG['aeid']}{SUFFIX}")
        if os.path.exists(file_path):
            os.remove(file_path)


def write_output(df):
    df = df[CONFIG['output_cols_filter']]
    df = get_metadata(df, CONFIG['aeid'])
    df = subset_chemicals(df)
    mb_value = f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
    print_(f"{status('computer_disk')} Writing output data to DB (~{mb_value})..")
    for col in ['conc', 'resp', 'fit_params']:
        df.loc[:, col] = df[col].apply(json.dumps)
    db_delete("output")
    db_append(df, "output")

    # Custom export
    file_path = os.path.join(OUT_DIR_PATH, f"{CONFIG['aeid']}{SUFFIX}")
    df[['dsstox_substance_id', 'hitcall']].to_parquet(file_path, compression='gzip')


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
    print(get_msg_with_elapsed_time(msg))


def subset_chemicals(dat):
    # A chemical id (chid) can have more than one sample id (spid). Use consensus hit (chit) logic: max
    dat = dat.sort_values(by=['chid', 'hitcall'], ascending=[True, False])
    # Drop duplicates based on 'chid' and keep the first occurrence (max 'hitcall' value)
    return dat.drop_duplicates(subset='chid', keep="first")


def get_metadata(df, aeid):
    chemical_df = get_chemical("spid", df["spid"].unique())
    df = pd.merge(chemical_df, df, on="spid", how="right")
    df['chid'].fillna(0, inplace=True)
    df['chid'] = df['chid'].astype(int)
    replacement_values = {
        'DMSO':  'DTXSID2021735',
        'Beta-Estradiol': 'DTXSID0020573',
    }

    index_mask = df['dsstox_substance_id'].isna()
    # Iterate through the DataFrame and replace masked values
    for index, row in df[index_mask].iterrows():
        spid_value = row['spid']
        replacement_value = replacement_values.get(spid_value)
        if replacement_value:
            df.loc[index, 'dsstox_substance_id'] = replacement_value
    return df


def get_chemical(field, spids):
    sample = 'sample'
    chemical = 'chemical'
    path_sample = os.path.join(INPUT_DIR_PATH, f"{sample}{SUFFIX}")
    path_chemical = os.path.join(INPUT_DIR_PATH, f"{chemical}{SUFFIX}")
    available = os.path.exists(path_sample) and os.path.exists(path_chemical)
    if not available:
        qstring = f"""
          SELECT spid, chid, dsstox_substance_id
          FROM {sample}
          LEFT JOIN {chemical} ON {chemical}.chid = {sample}.chid
          WHERE spid IN {str(tuple(spids))};
          """
        df = query_db(query=qstring)
    else:
        sample_df = pd.read_parquet(path_sample)
        chemical_df = pd.read_parquet(path_chemical)
        df = sample_df.merge(chemical_df, on='chid', how='left')
        df = df[df[field].isin(spids)]
        df = df[['spid', 'chid', 'dsstox_substance_id']]
    df = df.drop_duplicates()
    return df


def get_assay_component_endpoint(aeid):
    tbl = 'assay_component_endpoint'
    path = os.path.join(INPUT_DIR_PATH, f"{tbl}{SUFFIX}")
    if not os.path.exists(path):
        qstring = f"""
            SELECT aeid, assay_component_endpoint_name, normalized_data_type
            FROM {tbl} 
            WHERE aeid = {aeid};
            """
        df = query_db(query=qstring)
        return df
    else:
        df = pd.read_parquet(os.path.join(INPUT_DIR_PATH, f"{tbl}{SUFFIX}"))
        return df[df['aeid'] == aeid][['aeid', 'assay_component_endpoint_name', 'normalized_data_type']]


