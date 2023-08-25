import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from mysql import connector as mysql
from sqlalchemy import create_engine, text
from st_files_connection import FilesConnection

from src.pipeline.pipeline_constants import CONFIG_DIR_PATH, CONFIG_PATH, AEIDS_LIST_PATH, DDL_PATH, \
    DATA_DIR_PATH, LOG_DIR_PATH, RAW_DIR_PATH, METADATA_DIR_PATH, \
    CUTOFF_DIR_PATH, CUTOFF_TABLE, AEID_PATH, OUTPUT_DIR_PATH, METADATA_SUBSET_DIR_PATH, OUTPUT_COMPOUNDS_DIR_PATH
from src.pipeline.models.helper import get_mad

CONFIG = {}
logger = logging.getLogger(__name__)
START_TIME = 0
AEID = 0


def launch(config, config_path):
    """
    Initialize the processing instance, set up logging, and retrieve a list of assay endpoints to process.

    Args:
        config (dict): Configuration settings.
        config_path (str): Path to the configuration file.

    Returns:
        tuple: A tuple containing instance ID, total instances, assay endpoint ID list, and logger.
    """
    init_config(config)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--instance_id", type=int, help="Instance ID for workload distribution")
    parser.add_argument("-n", "--instances_total", type=int, help="Total number of instances")
    args = parser.parse_args()
    if args.instance_id is None or args.instances_total is None:
        print("Error: Please provide both --instance_id or -i and --instances_total or -n arguments")
        sys.exit(1)

    instance_id = args.instance_id
    instances_total = args.instances_total

    global START_TIME
    START_TIME = datetime.now()

    def create_empty_log_file(filename):
        with open(filename, 'w', encoding='utf-8'):
            pass

    log_filename = os.path.join(LOG_DIR_PATH, f"log_{instance_id}.log")
    error_filename = os.path.join(LOG_DIR_PATH, f"errors_{instance_id}.log")
    create_empty_log_file(log_filename)
    create_empty_log_file(error_filename)

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None, style='%'):
            super().__init__(fmt, datefmt, style)
            self.start_time = START_TIME

        def format(self, record):
            delta = datetime.now() - self.start_time
            hours, remainder = divmod(delta.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            hundredths = int((delta.microseconds / 10000) % 100)
            elapsed_time_formatted = f"[{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{hundredths:02}]"
            return f"{elapsed_time_formatted} {super().format(record)}"

    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    console_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = ElapsedTimeFormatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    aeid_list = get_partition(instance_id, instances_total)

    logger.info(f"🚀 Pytcpl launched!")
    logger.info(f"⚙️ Configuration located in {config_path}")
    logger.info(f"📜 Running pipeline for {len(aeid_list)} assay endpoints (See 'config/aeid_list.in')")
    logger.info(f"🧩 Engine: instance_id={instance_id}, instances_total={instances_total}\n")

    if CONFIG['enable_writing_db']:
        check_db()

    return instance_id, instances_total, aeid_list, logger


def prolog(new_aeid, instance_id):
    """
    Initialize processing for a specific assay endpoint.

    Args:
        new_aeid (int): New assay endpoint ID.
        instance_id (int): Instance ID for workload distribution.
    """
    init_aeid(new_aeid)
    with open(os.path.join(AEID_PATH, f'aeid_{instance_id}.in'), 'w') as f:
        f.write(str(AEID))
    logger.info(f"#-" * 50 + "\n")
    assay_component_endpoint_name = get_assay_info(AEID)['assay_component_endpoint_name']
    assay_info = f"{assay_component_endpoint_name} (aeid={AEID})"
    logger.info(f"🌱 Start processing new assay endpoint: {assay_info}")


def init_aeid(new_aeid):
    """
    Initialize the global assay endpoint identifier (AEID).

    This function sets the global AEID (assay endpoint identifier) to the provided value. The AEID is used to identify
    the specific assay endpoint being processed.

    Args:
        new_aeid (int): The new AEID value to set.

    Returns:
        None
    """
    global AEID
    AEID = int(new_aeid)


def epilog():
    """
    Log the completion of processing for an assay endpoint.

    This function is called to log the completion of processing for a specific assay endpoint. It can be used to
    indicate that the processing for a particular assay endpoint has finished.

    Returns:
        None
    """
    logger.info(f"🥕 Assay endpoint processing completed\n")


def bye():
    """
    Perform cleanup and logging as the pipeline completes.

    This function is called at the end of the pipeline to perform any necessary cleanup tasks and log
    completion messages.

    Returns:
        None
    """
    logger.info(f"🎊 Pipeline completed!")
    logger.info(f"👋 Goodbye")


def check_db():
    """
    Check the database schema for required tables and optionally drop new tables if enabled.

    This function checks whether the required database tables exist in the schema. It also provides the option
    to drop newly created tables if the configuration specifies enabling dropping new tables.

    Returns:
        None
    """
    if CONFIG['enable_dropping_all_new_tables']:
        query_db(f"DROP TABLE IF EXISTS {', '.join(CONFIG['new_db_tables'])};")
        logger.info(f"🧹 Dropped all relevant tables")

    for ddl_file in os.scandir(DDL_PATH):
        with open(ddl_file, 'r') as f:
            query_db(f.read())

    logger.info(f"👍 Verified the existence of required DB tables")


def get_assay_info(aeid):
    """
    Retrieve information about a specific assay endpoint.

    Args:
        aeid (int): Assay endpoint ID.

    Returns:
        dict: Dictionary containing assay endpoint information.
    """
    tbl_endpoint = 'assay_component_endpoint'
    tbl_component = 'assay_component'
    path_endpoint = os.path.join(METADATA_DIR_PATH, f"{tbl_endpoint}{CONFIG['file_format']}")
    path_component = os.path.join(METADATA_DIR_PATH, f"{tbl_component}{CONFIG['file_format']}")
    available = os.path.exists(path_endpoint) and os.path.exists(path_component)
    if not available or CONFIG['enable_allowing_reading_remote']:
        if CONFIG['enable_reading_db']:
            endpoint_query = f"SELECT * FROM {tbl_endpoint} WHERE aeid = {aeid};"
            acid = query_db(endpoint_query).iloc[0]['acid']
            component_query = f"SELECT * FROM {tbl_component} WHERE acid = {acid};"
            return pd.merge(query_db(endpoint_query), query_db(component_query), on='acid').iloc[0].to_dict()
        else:
            conn = st.experimental_connection('s3', type=FilesConnection)
            path_endpoint = f"{CONFIG['bucket']}/input/{tbl_endpoint}{CONFIG['file_format']}"
            path_component = f"{CONFIG['bucket']}/input/{tbl_component}{CONFIG['file_format']}"
            endpoint_df = conn.read(path_endpoint, input_format="parquet", ttl=600)
            component_df = conn.read(path_component, input_format="parquet", ttl=600)
    else:
        endpoint_df = pd.read_parquet(os.path.join(METADATA_DIR_PATH, f"{tbl_endpoint}{CONFIG['file_format']}"))
        component_df = pd.read_parquet(os.path.join(METADATA_DIR_PATH, f"{tbl_component}{CONFIG['file_format']}"))

    endpoint_df = endpoint_df[endpoint_df['aeid'] == aeid]
    df = pd.merge(endpoint_df, component_df, on='acid').iloc[0].to_dict()
    logger.debug(f"Read from {path_endpoint} and {path_endpoint}")
    return df


def fetch_raw_data():
    """
    Fetch raw data for a given assay endpoint and compute efficacy cutoffs.

    Returns:
        pandas.DataFrame: Processed DataFrame containing raw data.
    """
    logger.info(f"⏳ Fetching raw data..")
    path = os.path.join(RAW_DIR_PATH, f"{AEID}{CONFIG['file_format']}")
    if not os.path.exists(path):
        select_cols = ['spid', 'aeid', 'logc', 'resp', 'cndx', 'wllt']
        table_mapping = {'mc0.m0id': 'mc1.m0id', 'mc1.m0id': 'mc3.m0id'}
        join_string = ' AND '.join([f"{key} = {value}" for key, value in table_mapping.items()])
        qstring = f"SELECT {', '.join(select_cols)} " \
                  f"FROM mc0, mc1, mc3 " \
                  f"WHERE {join_string} AND aeid = {AEID};"
        df = query_db(query=qstring)
        df.to_parquet(path, compression='gzip')
    else:
        logger.debug(f"Read from {path}")
        df = pd.read_parquet(path)

    logger.info(f"ℹ️ Loaded {df.shape[0]} single datapoints")

    logger.info(f"🌡️ Computing efficacy cutoff")
    values = {}
    other_mthds = ['bmed.aeid.lowconc.twells', 'onesd.aeid.lowconc.twells']
    for mthd in load_method(lvl=4, aeid=AEID) + other_mthds:
        values.update({mthd: mc4_mthds(mthd, df)})

    bmad = values['bmad.aeid.lowconc.twells'] if 'bmad.aeid.lowconc.twells' in values else values[
        'bmad.aeid.lowconc.nwells']
    bmed = values['bmed.aeid.lowconc.twells']
    sd = values['onesd.aeid.lowconc.twells']

    cutoffs = [mc5_mthds(mthd, bmad) for mthd in load_method(lvl=5, aeid=AEID)]
    cutoff = max(list(filter(lambda item: item is not None, cutoffs)), default=0)
    out = pd.DataFrame({'aeid': [AEID], 'bmad': [bmad], 'bmed': [bmed], 'onesd': [sd], 'cutoff': [cutoff]})

    if CONFIG['enable_writing_db']:
        db_delete('cutoff')
        db_append(out, 'cutoff')

    path = os.path.join(CUTOFF_DIR_PATH, f"{AEID}{CONFIG['file_format']}")
    out.to_parquet(path)

    return df


def db_append(df, tbl):
    """
    Append data to a database table.

    Args:
        df (pandas.DataFrame): Data to append.
        tbl (str): Table name.
    """
    if CONFIG['enable_writing_db']:
        try:
            engine = get_sqlalchemy_engine()
            chunk_size = CONFIG['chunk_size']
            for start in range(0, len(df), chunk_size):
                chunk = df[start: start + chunk_size]
                chunk.to_sql(tbl, engine, if_exists='append', index=False)
            engine.dispose()
        except Exception as err:
            logger.error(err)

    folder_path = os.path.join(DATA_DIR_PATH, "all", tbl) if AEID == 0 else os.path.join(DATA_DIR_PATH, tbl)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{AEID}{CONFIG['file_format']}")
    df.to_parquet(file_path, compression='gzip')


def db_delete(tbl):
    """
    Delete data from a database table.

    Args:
        tbl (str): Table name.
    """
    if CONFIG['enable_writing_db']:
        query_db(f"DELETE FROM {tbl} WHERE aeid = {AEID};")

    file_path = os.path.join(DATA_DIR_PATH, tbl, f"{AEID}{CONFIG['file_format']}")
    if os.path.exists(file_path):
        os.remove(file_path)


def write_output(df):
    """
    Write output data to the database and custom output files.

    Args:
        df (pandas.DataFrame): DataFrame containing output data.
    """
    df = get_metadata(df)
    df = subset_chemicals(df)
    df = df[CONFIG['output_cols_filter']]
    mb_value = f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
    logger.info(f"💽 Writing output data to DB (~{mb_value})..")
    for col in ['conc', 'resp', 'fit_params']:
        df.loc[:, col] = df[col].apply(json.dumps)
    db_delete("output")
    db_append(df, "output")


def subset_chemicals(dat):
    """
    Subset chemicals based on consensus hit logic and keep the first occurrence.

    Args:
        dat (pandas.DataFrame): DataFrame containing data.

    Returns:
        pandas.DataFrame: Subsetted DataFrame.
    """
    # A chemical id (chid) can have more than one sample id (spid). Use consensus hit (chit) logic: max
    dat = dat.sort_values(by=['dsstox_substance_id', 'hitcall'], ascending=[True, False])
    # Drop duplicates based on 'chid' and keep the first occurrence (max 'hitcall' value)
    return dat.drop_duplicates(subset='dsstox_substance_id', keep="first")


def get_metadata(df):
    """
    Enhance the input DataFrame with additional metadata and replace missing chemical IDs.

    This function takes a DataFrame containing experimental data and enhances it by adding chemical
    information and replacing missing `dsstox_substance_id` values based on predefined replacement rules.

    Args:
        df (pandas.DataFrame): Input DataFrame containing experimental data.

    Returns:
        pandas.DataFrame: Enhanced DataFrame with added chemical information and replaced `dsstox_substance_id` values.
    """
    df = df.drop(columns=['dsstox_substance_id'])
    chemical_df = get_chemical(df["spid"].unique())
    df = pd.merge(chemical_df, df, on="spid", how="right")
    df['chid'].fillna(0, inplace=True)
    df['chid'] = df['chid'].astype(int)

    # Some SPIDs are not correctly handeld/mapped to DTXSID by DB and have synonyms
    replacement_values = {'DMSO': 'DTXSID2021735', 'Beta-Estradiol': 'DTXSID0020573',
                          'E2': 'DTXSID0020573'}
    index_mask = df['dsstox_substance_id'].isna()

    for index, row in df[index_mask].iterrows():
        spid_value = row['spid']
        if 'dmso' in spid_value.lower():
            replacement_value = replacement_values.get('DMSO')
            if replacement_value:
                df.loc[index, 'dsstox_substance_id'] = replacement_value
        elif 'beta-estradiol' in spid_value.lower():
            replacement_value = replacement_values.get('Beta-Estradiol')
            if replacement_value:
                df.loc[index, 'dsstox_substance_id'] = replacement_value
        else:
            replacement_value = replacement_values.get(spid_value)
            if replacement_value:
                df.loc[index, 'dsstox_substance_id'] = replacement_value
    return df


def get_chemical(spids):
    """
    Retrieve chemical information for a list of sample IDs.

    Args:
        spids (list): List of sample IDs.

    Returns:
        pandas.DataFrame: DataFrame containing chemical information for the specified sample IDs.
    """
    sample = 'sample'
    chemical = 'chemical'
    path_sample = os.path.join(METADATA_DIR_PATH, f"{sample}{CONFIG['file_format']}")
    path_chemical = os.path.join(METADATA_DIR_PATH, f"{chemical}{CONFIG['file_format']}")
    available = os.path.exists(path_sample) and os.path.exists(path_chemical)
    if not available or CONFIG['enable_allowing_reading_remote']:
        if CONFIG['enable_reading_db']:
            qstring = f"""
              SELECT spid, chid, dsstox_substance_id, chnm, casn
              FROM {sample}
              LEFT JOIN {chemical} ON {chemical}.chid = {sample}.chid
              WHERE spid IN {str(tuple(spids))};
              """
            return query_db(query=qstring).drop_duplicates()
        else:
            conn = st.experimental_connection('s3', type=FilesConnection)
            path_sample = f"{CONFIG['bucket']}/input/{sample}{CONFIG['file_format']}"
            path_chemical = f"{CONFIG['bucket']}/input/{chemical}{CONFIG['file_format']}"
            sample_df = conn.read(path_sample, input_format="parquet", ttl=600)
            chemical_df = conn.read(path_chemical, input_format="parquet", ttl=600)
    else:
        sample_df = pd.read_parquet(path_sample)
        chemical_df = pd.read_parquet(path_chemical)

    df = sample_df.merge(chemical_df, on='chid', how='left')
    df = df[df['spid'].isin(spids)]
    df = df[['spid', 'chid', 'dsstox_substance_id', 'chnm', 'casn']]
    df = df.drop_duplicates()
    logger.debug(f"Read from {path_sample} and {path_chemical}")
    return df


def get_assay_component_endpoint(aeid):
    """
    Retrieve assay component endpoint information for a specific assay endpoint.

    Args:
        aeid (int): Assay endpoint ID.

    Returns:
        pandas.DataFrame: DataFrame containing assay component endpoint information.
    """
    tbl = 'assay_component_endpoint'
    path = os.path.join(METADATA_DIR_PATH, f"{tbl}{CONFIG['file_format']}")
    if not os.path.exists(path) or CONFIG['enable_allowing_reading_remote']:
        if CONFIG['enable_reading_db']:
            qstring = f"""
                SELECT aeid, assay_component_endpoint_name, normalized_data_type
                FROM {tbl} 
                WHERE aeid = {aeid};
                """
            return query_db(query=qstring)
        else:
            conn = st.experimental_connection('s3', type=FilesConnection)
            path = f"{CONFIG['bucket']}/input/{tbl}{CONFIG['file_format']}"
            df = conn.read(path, input_format="parquet", ttl=600)
    else:
        df = pd.read_parquet(os.path.join(METADATA_DIR_PATH, f"{tbl}{CONFIG['file_format']}"))
    df = df[df['aeid'] == aeid][['aeid', 'assay_component_endpoint_name', 'normalized_data_type']]
    logger.debug(f"Read from {path}")
    return df


def get_cutoff(aeid=AEID):
    """
    Retrieve efficacy cutoff values for a specific assay endpoint.

    Returns:
        pandas.DataFrame: DataFrame containing cutoff values.
    """
    path = os.path.join(CUTOFF_DIR_PATH, f"{aeid}{CONFIG['file_format']}")
    if not os.path.exists(path) or CONFIG['enable_allowing_reading_remote']:
        if CONFIG['enable_reading_db']:
            qstring = f"""
                SELECT *
                FROM {CUTOFF_TABLE} 
                WHERE aeid = {aeid};
                """
            return query_db(query=qstring)
        else:
            conn = st.experimental_connection('s3', type=FilesConnection)
            path = f"{CONFIG['bucket']}/{CUTOFF_TABLE}/{aeid}{CONFIG['file_format']}"
            df = conn.read(path, input_format="parquet", ttl=600)
    else:
        df = pd.read_parquet(path)
    logger.debug(f"Read from {path}")
    return df


def mc4_mthds(mthd, df):
    """
    Compute specific metrics based on input data for level 4 methods.

    Args:
        mthd (str): Method name.
        df (pandas.DataFrame): Input data.

    Returns:
        float: Computed metric value.
    """
    cndx = df['cndx'].isin([1, 2])
    wllt_t = df['wllt'] == 't'
    mask = df.loc[cndx & wllt_t, 'resp']

    if mthd == 'bmad.aeid.lowconc.twells':
        return get_mad(mask)
    elif mthd == 'bmad.aeid.lowconc.nwells':
        return get_mad(df.loc[df['wllt'] == 'n', 'resp'])
    elif mthd == 'onesd.aeid.lowconc.twells':
        return mask.std()
    elif mthd == 'bmed.aeid.lowconc.twells':
        return mask.median()


def mc5_mthds(mthd, bmad):
    """
    Get predefined values for level 5 methods or perform computations using bmad.

    Args:
        mthd (str): Method name.
        bmad (float): Calculated bmad value.

    Returns:
        float: Method-specific value.
    """
    return {
        'pc20': 20,
        'pc50': 50,
        'pc70': 70,
        'log2_1.2': np.log2(1.2),
        'log10_1.2': np.log10(1.2),
        'log2_2': np.log2(2),
        'log10_2': np.log10(2),
        'neglog2_0.88': -1 * np.log2(0.88),
        'coff_2.32': 2.32,
        'fc0.2': 0.2,
        'fc0.3': 0.3,
        'fc0.5': 0.5,
        'pc05': 5,
        'pc10': 10,
        'pc25': 25,
        'pc30': 30,
        'pc95': 95,
        'bmad1': bmad,
        'bmad2': bmad * 2,
        'bmad3': bmad * 3,
        'bmad4': bmad * 4,
        'bmad5': bmad * 5,
        'bmad6': bmad * 6,
        'bmad10': bmad * 10,
        # 'maxmed20pct': lambda df: df['max_med'].aggregate(lambda x: np.max(x) * 0.20),  # is never used
    }.get(mthd)


def load_method(lvl, aeid):
    """
    Load and retrieve the list of methods for a specific level and assay endpoint.

    Args:
        lvl (int): Method level.
        aeid (int): Assay endpoint ID.

    Returns:
        list: List of method names.
    """
    tbl_aeid = f"mc{lvl}_aeid"
    tbl_methods = f"mc{lvl}_methods"
    path_aeid = os.path.join(METADATA_DIR_PATH, f"{tbl_aeid}{CONFIG['file_format']}")
    path_methods = os.path.join(METADATA_DIR_PATH, f"{tbl_methods}{CONFIG['file_format']}")
    available = os.path.exists(path_aeid) and os.path.exists(path_methods)
    if not available or CONFIG['enable_allowing_reading_remote']:
        if CONFIG['enable_reading_db']:
            flds = [f"b.mc{lvl}_mthd AS mthd"]
            tbls = [f"{tbl_aeid} AS a", f"{tbl_methods} AS b"]
            qstring = f"SELECT {', '.join(flds)} " \
                      f"FROM {', '.join(tbls)} " \
                      f"WHERE a.mc{lvl}_mthd_id = b.mc{lvl}_mthd_id " \
                      f"AND aeid = {aeid};"
            return query_db(query=qstring)["mthd"].tolist()
        else:
            conn = st.experimental_connection('s3', type=FilesConnection)
            path_aeid = f"{CONFIG['bucket']}/input/{tbl_aeid}{CONFIG['file_format']}"
            path_methods = f"{CONFIG['bucket']}/input/{tbl_methods}{CONFIG['file_format']}"
            df_aeid = conn.read(path_aeid, input_format="parquet", ttl=600)
            df_methods = conn.read(path_methods, input_format="parquet", ttl=600)

    else:
        df_aeid = pd.read_parquet(path_aeid)
        df_methods = pd.read_parquet(path_methods)

    df_aeid = df_aeid[df_aeid['aeid'] == aeid]
    level_col = f"mc{lvl}_mthd"
    df = df_aeid.merge(df_methods, left_on=f"mc{lvl}_mthd_id", right_on=f"mc{lvl}_mthd_id")
    df = df[level_col].tolist()
    logger.debug(f"Read from {path_aeid} and {path_methods}")
    return df


def load_config():
    """
    Load and retrieve the configuration settings.

    Returns:
        tuple: A tuple containing configuration settings and configuration directory path.
    """
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    return config, CONFIG_DIR_PATH


def init_config(config):
    """
    Initialize the global configuration variable.

    Args:
        config (dict): Configuration settings.
    """
    global CONFIG
    CONFIG = config


def get_partition(instance_id, instances_total):
    """
    Divide a list of assay endpoints into partitions for distributed processing.

    Args:
        instance_id (int): Instance ID for workload distribution.
        instances_total (int): Total number of instances.

    Returns:
        list: List of assay endpoint IDs for the given instance.
    """
    def read_aeids():
        with open(AEIDS_LIST_PATH, 'r') as file:
            ids_list = [line.strip() for line in file]
        return ids_list

    all_ids = read_aeids()
    partition_size = len(all_ids) // instances_total
    start_idx = instance_id * partition_size
    end_idx = start_idx + partition_size
    return all_ids[start_idx:end_idx]


def get_formatted_time_elapsed(start_time):
    """
    Get formatted time elapsed since a specific time.

    Args:
        start_time (datetime): Start time for elapsed time calculation.

    Returns:
        str: Formatted elapsed time string.
    """
    delta = datetime.now() - start_time
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    hundredths = int((delta.microseconds / 10000) % 100)
    elapsed_time_formatted = f"[{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{hundredths:02}]"
    return elapsed_time_formatted


def get_msg_with_elapsed_time(msg):
    """
    Add elapsed time to a message.

    Args:
        msg (str): Message.

    Returns:
        str: Message with elapsed time.
    """
    return f"{get_formatted_time_elapsed(START_TIME)} {msg}"


def merge_all_outputs():
    aeid_sorted = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_sorted{CONFIG['file_format']}"))
    output_paths = [os.path.join(OUTPUT_DIR_PATH, f"{aeid}{CONFIG['file_format']}") for aeid in aeid_sorted['aeid']]
    cutoff_paths = [os.path.join(CUTOFF_DIR_PATH, f"{aeid}{CONFIG['file_format']}") for aeid in aeid_sorted['aeid']]
    cols = ['dsstox_substance_id', 'aeid', 'hitcall']
    df_all = pd.concat([pd.read_parquet(file) for file in output_paths])
    cutoff_all = pd.concat([pd.read_parquet(file) for file in cutoff_paths])
    return df_all, cutoff_all


def get_db_config():
    """
    Retrieve database configuration parameters from a YAML file.

    Reads the database configuration file based on the user's login name and extracts the necessary parameters.

    Returns:
        tuple: A tuple containing the username, password, host, port, and database name for the MySQL connection.
    """
    with open(os.path.join(CONFIG_DIR_PATH, 'config_db.yaml'), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        login_name = os.getlogin()
        config_db = config[login_name]
        return config_db['username'], config_db['password'], config_db['host'], config_db['port'], config_db['db']


def get_sqlalchemy_engine():
    """
    Create a SQLAlchemy database engine.

    Constructs and returns an SQLAlchemy engine object using the retrieved database configuration parameters.

    Returns:
        sqlalchemy.engine.base.Engine: The SQLAlchemy database engine object.
    """
    username, password, host, port, db = get_db_config()
    url = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{db}"
    try:
        return create_engine(url)
    except Exception as error:
        print(f"Error connecting to MySQL: {error}")
        return None


def query_db(query):
    """
    Execute a SQL query on a MySQL database.

    Executes the provided SQL query on a MySQL database. If the query is a DELETE, CREATE, or DROP statement,
    it is executed using the mysql.connector library. For other queries, an SQLAlchemy engine is used, and the
    results are returned as a pandas DataFrame.

    Args:
        query (str): The SQL query to be executed.

    Returns:
        pandas.DataFrame or None: If the query is a SELECT statement, returns the query results as a DataFrame.
                                 Otherwise, returns None.
    """
    try:
        if any(query.lower().startswith(x) for x in ["delete", "create", "drop"]):
            user, pw, host, port, db = get_db_config()
            db_conn = mysql.connect(host=host, user=user, password=pw, port=port, database=db)
            cursor = db_conn.cursor()
            cursor.execute(query)
            db_conn.commit()
            db_conn.close()
        else:
            engine = get_sqlalchemy_engine()
            con = engine.connect()
            df = pd.read_sql(text(query), con=con)
            con.close()
            engine.dispose()
            return df
    except Exception as e:
        print(f"Error querying MySQL: {e}")
        return None


def get_output_data(aeid=AEID):
    """
    Retrieve output data for a specific AEID.

    This function fetches the output data for a given AEID. It determines whether to retrieve the data from a local file,
    a remote source, or a database query based on configuration settings. The retrieved data is returned as a dataframe.

    Returns:
        pd.DataFrame: DataFrame containing the output data for the specified AEID.

    """
    tbl = 'output'
    path = os.path.join(OUTPUT_DIR_PATH, f"{aeid}{CONFIG['file_format']}")
    if not os.path.exists(path) or CONFIG['enable_allowing_reading_remote']:
        if CONFIG['enable_reading_db']:
            qstring = f"SELECT * FROM {tbl} WHERE aeid = {aeid};"
            df = query_db(query=qstring)
        else:
            conn = st.experimental_connection('s3', type=FilesConnection)
            data_source = f"{CONFIG['bucket']}/{tbl}/{aeid}{CONFIG['file_format']}"
            df = conn.read(data_source, input_format="parquet", ttl=600)
    else:
        df = pd.read_parquet(path)
    length = df.shape[0]
    if length == 0:
        print(f"No data found for AEID {aeid}")
    print(f"{length} series loaded")
    return df


def get_output_compound(dsstox_substance_id):
    path = os.path.join(OUTPUT_COMPOUNDS_DIR_PATH, f"{dsstox_substance_id}{CONFIG['file_format']}")
    print(f"Fetch compound data: {dsstox_substance_id}..")
    df = pd.read_parquet(path)
    length = df.shape[0]
    if length == 0:
        print(f"No data found for compound: {dsstox_substance_id}")
    print(f"{length} series loaded")
    return df
