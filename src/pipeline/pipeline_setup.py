import time
from src.pipeline.pipeline_helper import load_config, init_config, init_aeid
from src.pipeline.pipeline_setup_helper import generate_balanced_aeid_list, \
    keep_viability_assay_endpoints_together, subset_candidate_assay_endoints_on_counts_and_hit_ratio, \
    export_metadata_tables_to_parquet, get_mechanistic_target_and_mode_of_action_annotations_from_ice, \
    get_all_related_assay_infos


def main():
    print("Started")
    start_time = time.time()
    
    config, _ = load_config()
    init_config(config)
    init_aeid(0)
    export_metadata_tables_to_parquet()
    get_mechanistic_target_and_mode_of_action_annotations_from_ice()
    df = subset_candidate_assay_endoints_on_counts_and_hit_ratio()
    df = keep_viability_assay_endpoints_together(config, df)
    generate_balanced_aeid_list(config, df)
    get_all_related_assay_infos(config)
    
    print("Finished")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
