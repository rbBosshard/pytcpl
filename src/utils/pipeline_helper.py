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
from st_files_connection import FilesConnection

from .constants import CONFIG_DIR_PATH, CONFIG_PATH, AEIDS_LIST_PATH, DDL_PATH, \
    EXPORT_DIR_PATH, LOG_DIR_PATH, RAW_DIR_PATH, CUSTOM_OUTPUT_DIR_PATH, INPUT_DIR_PATH, \
    CUTOFF_DIR_PATH, CUTOFF_TABLE, AEID_PATH
from .models.helper import get_mad
from .query_db import get_sqlalchemy_engine
from .query_db import query_db

CONFIG = {}
logger = logging.getLogger(__name__)
START_TIME = 0
AEID = 0


def launch(config, config_path):
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
    global AEID
    AEID = int(new_aeid)
    with open(os.path.join(AEID_PATH, f'aeid_{instance_id}.in'), 'w') as f:
        f.write(str(AEID))
    logger.info(f"#-" * 50 + "\n")
    assay_component_endpoint_name = get_assay_info(AEID)['assay_component_endpoint_name']
    assay_info = f"{assay_component_endpoint_name} (aeid={AEID})"
    logger.info(f"🌱 Start processing new assay endpoint: {assay_info}")


def epilog():
    logger.info(f"🥕 Assay endpoint processing completed\n")


def bye():
    logger.info(f"🎊 Pipeline completed!")
    logger.info(f"👋 Goodbye")


def check_db():
    if CONFIG['enable_dropping_all_new_tables']:
        query_db(f"DROP TABLE IF EXISTS {', '.join(CONFIG['new_db_tables'])};")
        logger.info(f"🧹 Dropped all relevant tables")

    for ddl_file in os.scandir(DDL_PATH):
        with open(ddl_file, 'r') as f:
            query_db(f.read())

    print(f"👍 Verified the existence of required DB tables")


def get_assay_info(aeid):
    tbl_endpoint = 'assay_component_endpoint'
    tbl_component = 'assay_component'
    path_endpoint = os.path.join(INPUT_DIR_PATH, f"{tbl_endpoint}{CONFIG['file_format']}")
    path_component = os.path.join(INPUT_DIR_PATH, f"{tbl_component}{CONFIG['file_format']}")
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
        endpoint_df = pd.read_parquet(os.path.join(INPUT_DIR_PATH, f"{tbl_endpoint}{CONFIG['file_format']}"))
        component_df = pd.read_parquet(os.path.join(INPUT_DIR_PATH, f"{tbl_component}{CONFIG['file_format']}"))

    endpoint_df = endpoint_df[endpoint_df['aeid'] == aeid]
    df = pd.merge(endpoint_df, component_df, on='acid').iloc[0].to_dict()
    logger.debug(f"Read from {path_endpoint} and {path_endpoint}")
    return df


def fetch_raw_data():
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


def db_append(dat, tbl):
    if CONFIG['enable_writing_db']:
        try:
            engine = get_sqlalchemy_engine()
            dat.to_sql(tbl, engine, if_exists='append', index=False)
        except Exception as err:
            logger.error(err)

    file_path = os.path.join(EXPORT_DIR_PATH, tbl, f"{AEID}{CONFIG['file_format']}")
    dat.to_parquet(file_path, compression='gzip')


def db_delete(tbl):
    if CONFIG['enable_writing_db']:
        query_db(f"DELETE FROM {tbl} WHERE aeid = {AEID};")

    file_path = os.path.join(EXPORT_DIR_PATH, tbl, f"{AEID}{CONFIG['file_format']}")
    if os.path.exists(file_path):
        os.remove(file_path)


def write_output(df):
    df = get_metadata(df)
    df = subset_chemicals(df)
    df = df[CONFIG['output_cols_filter']]
    mb_value = f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
    logger.info(f"💽 Writing output data to DB (~{mb_value})..")
    for col in ['conc', 'resp', 'fit_params']:
        df.loc[:, col] = df[col].apply(json.dumps)
    db_delete("output")
    db_append(df, "output")

    # Custom data
    file_path = os.path.join(CUSTOM_OUTPUT_DIR_PATH, f"{AEID}{CONFIG['file_format']}")
    df[['dsstox_substance_id', 'hitcall']].to_parquet(file_path, compression='gzip')


def subset_chemicals(dat):
    # A chemical id (chid) can have more than one sample id (spid). Use consensus hit (chit) logic: max
    dat = dat.sort_values(by=['dsstox_substance_id', 'hitcall'], ascending=[True, False])
    # Drop duplicates based on 'chid' and keep the first occurrence (max 'hitcall' value)
    return dat.drop_duplicates(subset='dsstox_substance_id', keep="first")


def get_metadata(df):
    df = df.drop(columns=['dsstox_substance_id'])
    chemical_df = get_chemical(df["spid"].unique())
    df = pd.merge(chemical_df, df, on="spid", how="right")
    df['chid'].fillna(0, inplace=True)
    df['chid'] = df['chid'].astype(int)
    replacement_values = {'DMSO': 'DTXSID2021735', 'Beta-Estradiol': 'DTXSID0020573',}
    index_mask = df['dsstox_substance_id'].isna()
    # Iterate through the DataFrame and replace masked values
    for index, row in df[index_mask].iterrows():
        spid_value = row['spid']
        replacement_value = replacement_values.get(spid_value)
        if replacement_value:
            df.loc[index, 'dsstox_substance_id'] = replacement_value
    return df


def get_chemical(spids):
    sample = 'sample'
    chemical = 'chemical'
    path_sample = os.path.join(INPUT_DIR_PATH, f"{sample}{CONFIG['file_format']}")
    path_chemical = os.path.join(INPUT_DIR_PATH, f"{chemical}{CONFIG['file_format']}")
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
    tbl = 'assay_component_endpoint'
    path = os.path.join(INPUT_DIR_PATH, f"{tbl}{CONFIG['file_format']}")
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
        df = pd.read_parquet(os.path.join(INPUT_DIR_PATH, f"{tbl}{CONFIG['file_format']}"))
    df = df[df['aeid'] == aeid][['aeid', 'assay_component_endpoint_name', 'normalized_data_type']]
    logger.debug(f"Read from {path}")
    return df


def get_cutoff():
    path = os.path.join(CUTOFF_DIR_PATH, f"{AEID}{CONFIG['file_format']}")
    if not os.path.exists(path) or  CONFIG['enable_allowing_reading_remote']:
        if CONFIG['enable_reading_db']:
            qstring = f"""
                SELECT bmad, bmed, onesd, cutoff
                FROM {CUTOFF_TABLE} 
                WHERE aeid = {AEID};
                """
            return query_db(query=qstring)
        else:
            conn = st.experimental_connection('s3', type=FilesConnection)
            path = f"{CONFIG['bucket']}/{CUTOFF_TABLE}/{AEID}{CONFIG['file_format']}"
            df = conn.read(path, input_format="parquet", ttl=600)
    else:
        df = pd.read_parquet(path)
    df = df[['bmad', 'bmed', 'onesd', 'cutoff']]
    logger.debug(f"Read from {path}")
    return df


def mc4_mthds(mthd, df):
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
    tbl_aeid = f"mc{lvl}_aeid"
    tbl_methods = f"mc{lvl}_methods"
    path_aeid = os.path.join(INPUT_DIR_PATH, f"{tbl_aeid}{CONFIG['file_format']}")
    path_methods = os.path.join(INPUT_DIR_PATH, f"{tbl_methods}{CONFIG['file_format']}")
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
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)
    return config, CONFIG_DIR_PATH


def init_config(config):
    global CONFIG
    CONFIG = config


def get_partition(instance_id, instances_total):
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
    delta = datetime.now() - start_time
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    hundredths = int((delta.microseconds / 10000) % 100)
    elapsed_time_formatted = f"[{int(hours):02}:{int(minutes):02}:{int(seconds):02}:{hundredths:02}]"
    return elapsed_time_formatted


def get_msg_with_elapsed_time(msg):
    return f"{get_formatted_time_elapsed(START_TIME)} {msg}"
