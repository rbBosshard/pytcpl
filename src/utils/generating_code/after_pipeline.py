import time

from src.pipeline.pipeline_helper import load_config, init_config, init_aeid
from src.utils.generating_code.after_pipeline_helper import process_chemical_data, check_all_cutoffs_available, \
    join_assay_tables


def main():
    print("Started")
    start_time = time.time()

    config, _ = load_config()
    init_config(config)
    init_aeid(0)

    check_all_cutoffs_available()
    process_chemical_data(config, max_workers=10)
    join_assay_tables(config)

    print("Finished")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
