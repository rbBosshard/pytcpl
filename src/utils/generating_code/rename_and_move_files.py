import os
import shutil
from src.utils.constants import DATA_DIR_PATH


def rename_and_move_files():
    """
    Renames and moves files in the specified directory.

    The script iterates through files in the source directory, renames them,
    and moves them to the destination directory.
    """
    dir = "raw"
    source_directory = os.path.join(DATA_DIR_PATH, dir)
    destination_directory = os.path.join(DATA_DIR_PATH, dir)

    for filename in os.listdir(source_directory):
        if filename.endswith(f'_{dir}.parquet.gzip'):
            id = filename.split('_')[0]
            new_filename = f'{id}.parquet.gzip'
            source_path = os.path.join(source_directory, filename)
            destination_path = os.path.join(destination_directory, new_filename)

            shutil.move(source_path, destination_path)
            print(f"Moved and renamed '{filename}' to '{new_filename}'")


if __name__ == "__main__":
    rename_and_move_files()
