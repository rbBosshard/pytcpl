import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import time

from src.pipeline.pipeline_helper import db_append, load_config, init_config, check_db, init_aeid
from src.utils.constants import OUTPUT_DIR_PATH, CHEMICAL_RESULT_DIR_PATH, METADATA_DIR_PATH, CUTOFF_DIR_PATH
from src.utils.query_db import get_db_config
import mysql.connector


# Ensure that in config/config.yaml: enable_writing_db: 1
def process_chemical_data(config, max_workers=10):
    """
    Process chemical data in parallel using ThreadPoolExecutor.

    Args:
        src_dir (str): Source directory containing Parquet files.
        dest_dir (str): Destination directory for processed Parquet files.
        max_workers (int): Maximum number of concurrent workers for parallel processing.
    """
    excepted = f"0{config['file_format']}"
    output_paths = [os.path.join(OUTPUT_DIR_PATH, file) for file in os.listdir(OUTPUT_DIR_PATH) if file != ".gitignore" and file != f"0{config['file_format']}"]
    cutoff_paths = [os.path.join(CUTOFF_DIR_PATH, file) for file in os.listdir(CUTOFF_DIR_PATH) if file != ".gitignore" and file != f"0{config['file_format']}"]

    cols = ['dsstox_substance_id', 'aeid', 'hitcall']
    df_all = pd.concat([pd.read_parquet(file) for file in output_paths])
    cutoff_all = pd.concat([pd.read_parquet(file) for file in cutoff_paths])

    # Can take several minutes...
    check_db()
    print("Append cutoff")
    # db_append(cutoff_all, 'cutoff')
    print("Append output")
    # db_append(df_all, 'output')

    unique_chemicals = df_all['dsstox_substance_id'].unique()
    aeids = df_all['aeid'].unique()  # Todo: Sort?
    aeids_strings = [str(x) for x in aeids]
    num_chemicals = len(unique_chemicals)
    num_aeids = len(aeids_strings) - 1  # aeid=0 is the df that contains ALL aeids
    print(f"Num aeids: {num_aeids}")
    print(f"Num compounds: {num_chemicals}")

    def process_chemical(i, chemical, df_all, dest_dir):
        print(f"{i+1}/{num_chemicals}: {chemical}")
        chemical_df = df_all[df_all['dsstox_substance_id'] == chemical]
        output_file = os.path.join(dest_dir, f'{chemical}.parquet.gzip')
        chemical_df.to_parquet(output_file, compression='gzip')

    with open(os.path.join(METADATA_DIR_PATH, 'unique_chemicals_tested.out'), 'w') as f:
        f.write('\n'.join(list(filter(lambda x: x is not None, unique_chemicals))))

    with open(os.path.join(METADATA_DIR_PATH, 'aeids.out'), 'w') as f:
        f.write('\n'.join(aeids_strings))


    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     for i, chemical in enumerate(unique_chemicals):
    #         executor.submit(process_chemical, i, chemical, df_all, CHEMICAL_RESULT_DIR_PATH)


def inspect_parquet(config):
    try:
        df_all = pd.read_parquet(os.path.join(OUTPUT_DIR_PATH, f"0{config['file_format']}"))
        df_cutoff = pd.read_parquet(os.path.join(CUTOFF_DIR_PATH, f"0{config['file_format']}"))
        start_time_ = time.time()
        test_assay_aeid = df_all[df_all['aeid'] == 2369]
        test_assay_dsstox_substance_id = df_all[df_all['dsstox_substance_id'] == 'DTXSID0020157']
        test_cutoff = df_cutoff[df_cutoff['aeid'] == 2369]
        print(f"Execution time: {time.time() - start_time_:.2f} seconds")
        print(len(test_assay_aeid))
        print(len(test_assay_dsstox_substance_id))
        print(len(test_cutoff))
    except Exception as e:
        print(f"Error: {e}")


def check_all_cutoffs_available():
    output_paths = [file for file in os.listdir(OUTPUT_DIR_PATH)]
    cutoff_paths = [file for file in os.listdir(CUTOFF_DIR_PATH)]
    if all(elem in cutoff_paths for elem in output_paths):
        print("cutoff_paths contains all elements of output_paths")
    else:
        print("cutoff_paths contains not all elements of output_paths")
        exit()


if __name__ == "__main__":

    print("Started")
    start_time = time.time()
    config, _ = load_config()
    init_config(config)
    init_aeid(0)
    max_workers = 10

    # check_all_cutoffs_available
    # process_chemical_data(config, max_workers)
    inspect_parquet(config)

    print("Finished")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
