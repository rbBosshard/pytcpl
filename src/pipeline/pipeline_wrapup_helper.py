import os

import pandas as pd

from src.pipeline.pipeline_helper import merge_all_outputs, check_db, db_append
from src.pipeline.pipeline_constants import METADATA_SUBSET_DIR_PATH, METADATA_DIR_PATH, OUTPUT_DIR_PATH, \
    CUTOFF_DIR_PATH, AEIDS_LIST_PATH


def remove_files_not_matching_to_aeid_list():
    directories = [OUTPUT_DIR_PATH, CUTOFF_DIR_PATH]

    with open(AEIDS_LIST_PATH, 'r') as f:
        ids = set(line.strip() for line in f)

    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith('.parquet.gzip'):
                id = filename.replace('.parquet.gzip', '')
                if id not in ids:
                    file_path = os.path.join(directory, filename)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    else:
                        print(f"Warning: File not found: {file_path} (aeid: {id})")


def merge_all_results(config):

    df_all, cutoff_all = merge_all_outputs()

    print("Can take several minutes..")
    check_db()  # For writing to DB ensure it holds that in config/config.yaml: enable_writing_db: 1
    db_append(cutoff_all, 'cutoff')
    db_append(df_all, 'output')

    compounds = df_all['dsstox_substance_id'].dropna().unique()
    compounds_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"compounds_tested{config['file_format']}")
    pd.DataFrame(compounds, columns=['dsstox_substance_id']).to_parquet(compounds_path, compression='gzip')

    num_compounds = len(compounds)
    print(f"Num compounds: {num_compounds}")

    with open(os.path.join(METADATA_DIR_PATH, 'unique_chemicals_tested.out'), 'w') as f:
        for chemical in compounds:
            f.write(str(chemical) + '\n')




