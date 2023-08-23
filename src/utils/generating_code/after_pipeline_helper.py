import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.pipeline.pipeline_helper import merge_all_outputs, check_db, db_append
from src.utils.constants import METADATA_SUBSET_DIR_PATH, METADATA_DIR_PATH, CHEMICAL_RESULT_DIR_PATH, OUTPUT_DIR_PATH, \
    CUTOFF_DIR_PATH

ENABLE_SEPARATE_CHEMICAL_RESULTS = 0


def process_chemical_data(config, max_workers=10):

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

    if ENABLE_SEPARATE_CHEMICAL_RESULTS:
        get_chemical_results(df_all, max_workers, num_compounds, compounds)


def get_chemical_results(df_all, max_workers, num_chemicals, unique_chemicals):
    def process_chemical(i, chemical, num_chemicals, df_all, dest_dir):
        print(f"{i + 1}/{num_chemicals}: {chemical}")
        chemical_df = df_all[df_all['dsstox_substance_id'] == chemical]
        output_file = os.path.join(dest_dir, f'{chemical}.parquet.gzip')
        chemical_df.to_parquet(output_file, compression='gzip')

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, chemical in enumerate(unique_chemicals):
            executor.submit(process_chemical, i, chemical, num_chemicals, df_all, CHEMICAL_RESULT_DIR_PATH)


def check_all_cutoffs_available():
    output_paths = [file for file in os.listdir(OUTPUT_DIR_PATH)]
    cutoff_paths = [file for file in os.listdir(CUTOFF_DIR_PATH)]
    if all(elem in cutoff_paths for elem in output_paths):
        pass
    else:
        print("cutoff_paths contains not all elements of output_paths")


def join_assay_tables(config):
    def convert_to_json_serializable(value):
        if isinstance(value, np.integer):
            return int(value)
        elif pd.isna(value):
            return None
        else:
            return value
    def save_distinct_values(column, values):
        out_file_path = os.path.join(METADATA_SUBSET_DIR_PATH, "assay_info_distinct_values", f"{column}.out")
        with open(out_file_path, 'w') as out_file:
            out_file.write('\n'.join(map(str, values)))

    os.makedirs(os.path.join(METADATA_SUBSET_DIR_PATH, "assay_info_distinct_values"), exist_ok=True)

    df_aeids = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids{config['file_format']}"))
    mechanistic_target_and_mode_of_action_df = pd.read_parquet(
        os.path.join(METADATA_DIR_PATH, f"mechanistic_target_and_mode_of_action{config['file_format']}"))
    assay_component_df = pd.read_parquet(os.path.join(METADATA_DIR_PATH, f"assay_component{config['file_format']}"))
    assay_component_endpoint_df = pd.read_parquet(
        os.path.join(METADATA_DIR_PATH, f"assay_component_endpoint{config['file_format']}"))

    assay_info_df = (
        assay_component_df.merge(assay_component_endpoint_df, on='acid')
        .merge(df_aeids[['aeid']], left_on='aeid', right_on='aeid')
        .merge(mechanistic_target_and_mode_of_action_df,
               left_on='assay_component_name', right_on='new_AssayEndpointName', how='left')
    )

    assay_info_distinct_values = {}
    with open(os.path.join(METADATA_SUBSET_DIR_PATH, "assay_info_distinct_values_counts.out"), 'w') as length_file:
        for column in assay_info_df.columns:
            distinct_values = assay_info_df[column].unique()

            # Convert ndarray to list and handle NaN values
            if isinstance(distinct_values[0], np.ndarray):
                distinct_values = [list(val) if isinstance(val, np.ndarray) else val for val in distinct_values]
            else:
                distinct_values = [convert_to_json_serializable(val) for val in distinct_values]

            assay_info_distinct_values[column] = distinct_values
            save_distinct_values(column, distinct_values)
            length_file.write(f"{column}: {len(distinct_values)}):\n")

    json_file_path = os.path.join(METADATA_SUBSET_DIR_PATH, "assay_info_distinct_values.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(assay_info_distinct_values, json_file)

    assay_info_df = assay_info_df.drop_duplicates(subset='aeid', keep='first')
    assay_info_df.to_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"assay_info{config['file_format']}"),
                             compression='gzip')


    return assay_info_df, assay_info_distinct_values

