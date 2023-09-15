import json
import time

import sys
import os
import pandas as pd

from src.pipeline.pipeline_constants import METADATA_SUBSET_DIR_PATH, METADATA_DIR_PATH, FILE_FORMAT

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

from src.pipeline.pipeline_helper import load_config, init_config, init_aeid
from src.pipeline.pipeline_setup_helper import generate_balanced_aeid_list, \
    keep_viability_assay_endpoints_together, subset_for_candidate_assay_endpoints, \
    export_metadata_tables_to_parquet, get_mechanistic_target_and_mode_of_action_annotations_from_ice, \
    get_all_related_assay_infos


def adapt_viability_counterparts(target_df, viability_df):
    viability_counterparts_name = {}
    viability_counterparts_aeid = {}
    for _, row in target_df.iterrows():
        # Account for lower/uppercase mismatch, e.g. ('TOX21_VDR_BLA_agonist_ratio' & 'TOX21_VDR_BLA_Agonist_viability')
        assay_name = row['assay_component_endpoint_name'].lower()
        aeid = str(row['aeid'])
        # Check if the assay name ends with _ratio or another name and append '_viability' as counterpart
        if assay_name.endswith('_ratio'):
            counterpart_name = assay_name[:-6] + '_viability'
        else:
            counterpart_name = assay_name + '_viability'

        counterpart_name = counterpart_name.lower()

        if counterpart_name in viability_df['assay_component_endpoint_name'].str.lower().values:
            viability_counterparts_name[row['assay_component_endpoint_name']] = \
                viability_df.loc[viability_df['assay_component_endpoint_name'].str.lower() == counterpart_name, 'assay_component_endpoint_name'].values[0]
            viability_counterparts_aeid[aeid] = str(viability_df.loc[viability_df['assay_component_endpoint_name'].str.lower() == counterpart_name, 'aeid'].values[0])

    json_file_path = os.path.join(METADATA_SUBSET_DIR_PATH, "viability_counterparts_name.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(viability_counterparts_name, json_file)

    json_file_path = os.path.join(METADATA_SUBSET_DIR_PATH, "viability_counterparts_aeid.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(viability_counterparts_aeid, json_file)

    mask = viability_df['assay_component_endpoint_name'].isin(set(viability_counterparts_name.values()))
    viability_df = viability_df[mask]

    return viability_df


def split_assays(df):
    viability_assay_endpoints = df[df['assay_component_endpoint_name'].str.endswith('viability')]
    target_assay_endpoints = df[~df['assay_component_endpoint_name'].str.endswith('viability')]
    return target_assay_endpoints, viability_assay_endpoints


def filter_on_count_and_lower_hitcall_ratio(config, target_assay_endpoints):
    assays_to_drop = set()

    for _, row in target_assay_endpoints.iterrows():
        count = row['count']
        ratio = row['ratio']
        name = row['assay_component_endpoint_name']
        if count < config['threshold_subsetting_aeids_on_count_compounds_tested'] or ratio < config['threshold_subsetting_aeids_on_hit_ratio']:
            assays_to_drop.add(name)

    target_assay_endpoints = target_assay_endpoints.drop(target_assay_endpoints[target_assay_endpoints['assay_component_endpoint_name'].isin(assays_to_drop)].index)
    return target_assay_endpoints


def include_burst_assays(df):
    path = os.path.join(METADATA_DIR_PATH, f"assay_component_endpoint{FILE_FORMAT}")
    df = pd.read_parquet(path)
    burst_assays = df[df['burst_assay'] == 1]

    df = pd.concat([df, burst_assays], ignore_index=True)



def main():
    print("Started")
    start_time = time.time()
    
    config, _ = load_config()
    init_config(config)
    init_aeid(0)
    # export_metadata_tables_to_parquet()
    # ice_df = get_mechanistic_target_and_mode_of_action_annotations_from_ice()
    df = subset_for_candidate_assay_endpoints()
    df = handle_viability_assays(config, df)
    generate_balanced_aeid_list(config, df)
    get_all_related_assay_infos(config)
    
    print("Finished")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


def handle_viability_assays(config, df):
    is_burst_assay = df['burst_assay'] == 1
    burst_assays = df[is_burst_assay]
    aeids_burst_assays = df[is_burst_assay]['aeid']
    df_without_burst_assays = df[~is_burst_assay]
    target_assay_endpoints = df_without_burst_assays[~df_without_burst_assays['assay_component_endpoint_name'].str.endswith('viability')]
    target_assay_endpoints = filter_on_count_and_lower_hitcall_ratio(config, target_assay_endpoints)
    aeids_target_assays = target_assay_endpoints['aeid']
    viability_assay_endpoints = df[df['assay_component_endpoint_name'].str.endswith('viability')]
    viability_assay_endpoints = adapt_viability_counterparts(target_assay_endpoints, viability_assay_endpoints)
    aeids_target_viability_assays = viability_assay_endpoints['aeid']
    df = pd.concat([burst_assays, target_assay_endpoints, viability_assay_endpoints], ignore_index=True)
    df = df.drop_duplicates().reset_index()

    destination_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_burst_assays{FILE_FORMAT}")
    export_df = pd.DataFrame({'aeid': aeids_burst_assays})
    export_df.to_parquet(destination_path, compression='gzip')
    destination_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_burst_assays.csv")
    export_df.to_csv(destination_path, index=False)

    destination_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_target_assays{FILE_FORMAT}")
    export_df = pd.DataFrame({'aeid': aeids_target_assays})
    export_df.to_parquet(destination_path, compression='gzip')
    destination_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_target_assays.csv")
    export_df.to_csv(destination_path, index=False)

    destination_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_target_viability_assays{FILE_FORMAT}")
    export_df = pd.DataFrame({'aeid': aeids_target_viability_assays})
    export_df.to_parquet(destination_path, compression='gzip')
    destination_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_target_viability_assays.csv")
    export_df.to_csv(destination_path, index=False)

    return df


if __name__ == "__main__":
    main()
