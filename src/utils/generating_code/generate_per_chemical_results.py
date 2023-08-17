import os

import pandas as pd
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor


from src.utils.constants import OUTPUT_DIR_PATH, CHEMICAL_RESULT_DIR_PATH

src_dir = OUTPUT_DIR_PATH
dest_dir = CHEMICAL_RESULT_DIR_PATH
paths = [os.path.join(src_dir, file) for file in os.listdir(src_dir) if file != ".gitignore"]

cols = ['dsstox_substance_id', 'aeid', 'hitcall']
all_results_df = pd.concat([pd.read_parquet(file)[cols] for file in paths])
unique_chemicals = all_results_df['dsstox_substance_id'].unique()

num_chemicals = len(unique_chemicals)


def process_chemical(i, chemical, all_results_df, dest_dir):
    print(f"{i}/{num_chemicals}: {chemical}")
    chemical_df = all_results_df[all_results_df['dsstox_substance_id'] == chemical]
    output_file = os.path.join(dest_dir, f'{chemical}.parquet.gzip')
    chemical_df.to_parquet(output_file, compression='gzip')


# Using ThreadPoolExecutor to parallelize processing
with ThreadPoolExecutor(max_workers=10) as executor:
    unique_chemicals = all_results_df['dsstox_substance_id'].unique()
    for i, chemical in enumerate(unique_chemicals):
        executor.submit(process_chemical, i, chemical, all_results_df, dest_dir)
