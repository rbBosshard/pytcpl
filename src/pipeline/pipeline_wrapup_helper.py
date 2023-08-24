import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from src.pipeline.pipeline_helper import merge_all_outputs, check_db, db_append
from src.pipeline.pipeline_constants import METADATA_SUBSET_DIR_PATH, METADATA_DIR_PATH, OUTPUT_DIR_PATH, \
    CUTOFF_DIR_PATH, AEIDS_LIST_PATH, OUTPUT_COMPOUNDS_DIR_PATH


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


def save_all_results(config, df_all, cutoff_all):
    print("Wait for it..")
    # For writing to DB ensure it holds that in config/config.yaml: enable_writing_db: 1
    check_db()
    print("Takes approx. 10 minutes depending on CPUs")
    db_append(cutoff_all, 'cutoff')
    db_append(df_all, 'output')

    compounds = df_all['dsstox_substance_id'].dropna().unique()
    compounds_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"compounds_tested{config['file_format']}")
    pd.DataFrame(compounds, columns=['dsstox_substance_id']).to_parquet(compounds_path, compression='gzip')

    num_compounds = len(compounds)
    print(f"Number of compounds tested over all assay endpoints: {num_compounds}")

    with open(os.path.join(METADATA_DIR_PATH, 'unique_chemicals_tested.out'), 'w') as f:
        for chemical in compounds:
            f.write(str(chemical) + '\n')

    if config['enable_output_compounds']:
        get_chemical_results(df_all, num_compounds, compounds)

    return df_all, cutoff_all


def get_chemical_results(df_all, num_chemicals, unique_chemicals):
    def process_chemical(i, chemical):
        print(f"{i + 1}/{num_chemicals}: {chemical}")
        chemical_df = df_all[df_all['dsstox_substance_id'] == chemical]
        output_file = os.path.join(OUTPUT_COMPOUNDS_DIR_PATH, f'{chemical}.parquet.gzip')
        chemical_df.to_parquet(output_file, compression='gzip')

    os.makedirs(OUTPUT_COMPOUNDS_DIR_PATH, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(os.cpu_count() * 2 - 1, 1)) as executor:
        for i, chemical in enumerate(unique_chemicals):
            executor.submit(process_chemical, i, chemical)
