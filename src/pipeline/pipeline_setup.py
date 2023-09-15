import time

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

from src.pipeline.pipeline_helper import load_config, init_config, init_aeid
from src.pipeline.pipeline_setup_helper import generate_balanced_aeid_list, \
    subset_for_candidate_assay_endpoints, \
    export_metadata_tables_to_parquet, get_mechanistic_target_and_mode_of_action_annotations_from_ice, \
    get_all_related_assay_infos, get_chemical_qc, handle_viability_assays


def main():
    print("Started")
    start_time = time.time()
    
    config, _ = load_config()
    init_config(config)
    init_aeid(0)
    export_metadata_tables_to_parquet()
    get_mechanistic_target_and_mode_of_action_annotations_from_ice()
    get_chemical_qc()
    df = subset_for_candidate_assay_endpoints()
    df = handle_viability_assays(config, df)
    generate_balanced_aeid_list(config, df)
    get_all_related_assay_infos(config)
    
    print("Finished")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
