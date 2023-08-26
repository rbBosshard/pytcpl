import os
import sys
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

from src.pipeline.pipeline_helper import load_config, init_config, init_aeid, merge_all_outputs
from src.pipeline.pipeline_wrapup_helper import save_all_results, remove_files_not_matching_to_aeid_list


def main():
    print("Started")
    start_time = time.time()

    config, _ = load_config()
    init_config(config)
    init_aeid(0)

    remove_files_not_matching_to_aeid_list()
    df_all, cutoff_all = merge_all_outputs()
    save_all_results(config, df_all, cutoff_all)

    print("Finished")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
