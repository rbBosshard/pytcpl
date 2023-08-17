import os
import shutil
from src.utils.constants import EXPORT_DIR_PATH

dir = "raw"
# Directory where the original files are located
source_directory = os.path.join(EXPORT_DIR_PATH, dir)

# Directory where the renamed files will be moved
destination_directory = os.path.join(EXPORT_DIR_PATH, dir)

# Iterate through all files in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith(f'_{dir}.parquet.gzip'):
        # Extract the id from the filename
        id = filename.split('_')[0]

        # New filename after renaming
        new_filename = f'{id}.parquet.gzip'

        # Full paths to the source and destination files
        source_path = os.path.join(source_directory, filename)
        destination_path = os.path.join(destination_directory, new_filename)

        # Rename and move the file
        shutil.move(source_path, destination_path)
        print(f"Moved and renamed '{filename}' to '{new_filename}'")
