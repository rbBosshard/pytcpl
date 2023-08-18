import os

from src.utils.constants import OUTPUT_DIR_PATH, CUSTOM_OUTPUT_DIR_PATH, AEIDS_LIST_PATH

directories = [OUTPUT_DIR_PATH, CUSTOM_OUTPUT_DIR_PATH]

with open(AEIDS_LIST_PATH, 'r') as f:
    ids = set(line.strip() for line in f)

for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith('.parquet.gzip'):
            id = filename.replace('.parquet.gzip', '')
            if id not in ids:
                file_path = os.path.join(directory, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")



