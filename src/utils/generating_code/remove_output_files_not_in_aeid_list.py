import os
from src.utils.constants import OUTPUT_DIR_PATH, CUSTOM_OUTPUT_DIR_PATH, AEIDS_LIST_PATH


def main():
    """
    Deletes parquet files from specified directories that are not in the AEID list.

    Reads AEID list from AEIDS_LIST_PATH and deletes parquet files from OUTPUT_DIR_PATH
    and CUSTOM_OUTPUT_DIR_PATH if their corresponding AEIDs are not in the list.
    """
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


if __name__ == "__main__":
    main()
