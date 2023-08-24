import time

from src.pipeline.pipeline_helper import load_config, init_config, init_aeid
from src.pipeline.pipeline_wrapup_helper import merge_all_results, remove_files_not_matching_to_aeid_list


def main():
    print("Started")
    start_time = time.time()

    config, _ = load_config()
    init_config(config)
    init_aeid(0)

    remove_files_not_matching_to_aeid_list()
    merge_all_results(config)

    print("Finished")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
