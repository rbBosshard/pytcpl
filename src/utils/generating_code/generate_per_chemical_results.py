import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from src.utils.constants import OUTPUT_DIR_PATH, CHEMICAL_RESULT_DIR_PATH, METADATA_DIR_PATH

def process_chemical_data(src_dir, dest_dir, max_workers=10):
    """
    Process chemical data in parallel using ThreadPoolExecutor.

    Args:
        src_dir (str): Source directory containing Parquet files.
        dest_dir (str): Destination directory for processed Parquet files.
        max_workers (int): Maximum number of concurrent workers for parallel processing.
    """
    paths = [os.path.join(src_dir, file) for file in os.listdir(src_dir) if file != ".gitignore"]
    cols = ['dsstox_substance_id', 'aeid', 'hitcall']
    all_results_df = pd.concat([pd.read_parquet(file)[cols] for file in paths])
    unique_chemicals = all_results_df['dsstox_substance_id'].unique()
    num_chemicals = len(unique_chemicals)

    def process_chemical(i, chemical, all_results_df, dest_dir):
        print(f"{i+1}/{num_chemicals}: {chemical}")
        chemical_df = all_results_df[all_results_df['dsstox_substance_id'] == chemical]
        output_file = os.path.join(dest_dir, f'{chemical}.parquet.gzip')
        chemical_df.to_parquet(output_file, compression='gzip')

    with open(os.path.join(METADATA_DIR_PATH, 'unique_chemicals_tested.out'), 'w') as f:
        f.write('\n'.join(list(filter(lambda x: x is not None, unique_chemicals))))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, chemical in enumerate(unique_chemicals):
            executor.submit(process_chemical, i, chemical, all_results_df, dest_dir)

if __name__ == "__main__":
    src_directory = OUTPUT_DIR_PATH
    destination_directory = CHEMICAL_RESULT_DIR_PATH
    process_chemical_data(src_directory, destination_directory)
