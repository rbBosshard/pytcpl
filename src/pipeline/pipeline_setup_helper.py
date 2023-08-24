import os
import numpy as np
import pandas as pd
import json

from src.pipeline.pipeline_constants import METADATA_DIR_PATH, AEIDS_LIST_PATH, METADATA_SUBSET_DIR_PATH
from src.pipeline.pipeline_helper import query_db

pd.set_option('mode.chained_assignment', None)

INSTANCES_TOTAL = 4
THRESHOLD_COMPOUNDS_TESTED = 2000
THRESHOLD_HIT_RATIO = 0.005


def save_aeids(config, df):
    aeids = df['aeid'].unique()
    aeids_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids{config['file_format']}")
    pd.DataFrame(aeids, columns=['aeid']).to_parquet(aeids_path, compression='gzip')

    aeids_sorted = np.sort(aeids)
    aeids_sorted_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_sorted{config['file_format']}")
    pd.DataFrame(aeids_sorted, columns=['aeid']).to_parquet(aeids_sorted_path, compression='gzip')
    with open(os.path.join(METADATA_SUBSET_DIR_PATH, 'aeids.out'), 'w') as f:
        for aeid in aeids:
            f.write(str(aeid) + '\n')
    with open(os.path.join(METADATA_SUBSET_DIR_PATH, 'aeids_sorted.out'), 'w') as f:
        for aeid in aeids_sorted:
            f.write(str(aeid) + '\n')
    return aeids


def generate_balanced_aeid_list(config, df):
    df = df.sort_values('hitc_1_count', ascending=False).reset_index(drop=True)
    aeids = save_aeids(config, df)
    num_aeids = len(aeids)
    tasks_per_instance = (len(aeids) + INSTANCES_TOTAL - 1) // INSTANCES_TOTAL
    distributed_tasks = distribute_aeids_to_instances(aeids, INSTANCES_TOTAL)
    with open(AEIDS_LIST_PATH, "w") as file:
        for i, instance_tasks in enumerate(distributed_tasks):
            for task_id in instance_tasks[:tasks_per_instance]:
                file.write(str(task_id) + "\n")
    print(f"Instances total: {INSTANCES_TOTAL}")
    print(f"Total num aeids to process: {num_aeids}")
    print(f"Num aeids per instance to process: {tasks_per_instance}")


def keep_viability_assay_endpoints_together(df):
    # Separate dataframes based on endpoint names
    ratio_df = df[
        df['assay_component_endpoint_name'].str.endswith("_ratio") & (df['count'] > THRESHOLD_COMPOUNDS_TESTED) & (
                df['ratio'] > THRESHOLD_HIT_RATIO)]
    viability_df = df[df['assay_component_endpoint_name'].str.endswith("_viability")]
    filtered_df = df[~df['assay_component_endpoint_name'].str.endswith(("_ratio", "_viability", "_ch1", "_ch2")) & (
            df['count'] > THRESHOLD_COMPOUNDS_TESTED) & (df['ratio'] > THRESHOLD_HIT_RATIO)]
    # Create prefixes and match rows
    ratio_df['pre'] = ratio_df['assay_component_endpoint_name'].str.replace('_ratio', '')
    viability_df['pre'] = viability_df['assay_component_endpoint_name'].str.replace('_viability', '')
    matching = viability_df[viability_df['pre'].isin(ratio_df['pre'])]
    # Concatenate dataframes
    ratio_viability_df = pd.concat([ratio_df.drop(columns=['pre']), matching.drop(columns=['pre'])], ignore_index=True)
    # Match and concatenate for filtered_df
    filtered_df['pre'] = filtered_df['assay_component_endpoint_name']
    matching = viability_df[viability_df['pre'].isin(filtered_df['pre'])]
    endpoint_viability_df = pd.concat([filtered_df.drop(columns=['pre']), matching.drop(columns=['pre'])],
                                      ignore_index=True)
    # Final combined dataframe
    df = pd.concat([endpoint_viability_df, ratio_viability_df], ignore_index=True)
    return df


def subset_candidate_assay_endoints_on_counts_and_hit_ratio():
    os.makedirs(METADATA_SUBSET_DIR_PATH, exist_ok=True)
    dest_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"candidate_assay_endpoints.parquet.gzip")
    if not os.path.exists(dest_path):
        query = f"SELECT aeid, " \
                f"COUNT(*) as count, " \
                f"SUM(hitc = 1) AS hitc_1_count, " \
                f"SUM(hitc = 0) AS hitc_0_count, " \
                f"SUM(hitc = 1) / COUNT(*) AS ratio " \
                f"FROM invitrodb_v3o5.mc5 " \
                f"GROUP BY aeid;"
        df_counts = query_db(query)

        query = f"SELECT aeid, " \
                f"assay_component_endpoint_name, " \
                f"analysis_direction," \
                f"signal_direction " \
                f"FROM invitrodb_v3o5.assay_component_endpoint " \
                f"WHERE analysis_direction='positive' " \
            # f"AND signal_direction='gain';"
        df_analysis_direction = query_db(query)

        df = df_counts.merge(df_analysis_direction, on="aeid", how="inner")
        df.to_parquet(dest_path, compression='gzip')
    else:
        df = pd.read_parquet(dest_path)
    return df


def distribute_aeids_to_instances(tasks, total_instances):
    """
    Distributes AEIDs (Assay Endpoint IDs) to instances for parallel processing.

    Args:
        tasks (list): List of AEIDs to be distributed.
        total_instances (int): Total number of instances for parallel processing.

    Returns:
        list of lists: Distributed AEID tasks for each instance.
    """
    distributed_tasks = [[] for _ in range(total_instances)]
    for i, task_id in enumerate(tasks):
        worker_idx = i % total_instances
        distributed_tasks[worker_idx].append(task_id)
    return distributed_tasks


def export_metadata_tables_to_parquet():
    """
    Export data from specified MySQL tables to Parquet files.
    """
    tables = ["assay_component", "assay_component_endpoint",
              "assay_component_endpoint_descriptions",
              "sample", "chemical",
              "mc4_aeid", "mc5_aeid",
              "mc4_methods", "mc5_methods"]

    for table in tables:
        df = query_db(f'SELECT * FROM {table}')
        destination_path = os.path.join(METADATA_DIR_PATH, f"{table}.parquet.gzip")
        df.to_parquet(destination_path, compression='gzip')


def get_mechanistic_target_and_mode_of_action_annotations_from_ice():
    # Download link: "https://ice.ntp.niehs.nih.gov/downloads/MOA/cHTSMT_ALL.xlsx"
    src_path = os.path.join(METADATA_DIR_PATH, f"cHTSMT_ALL.xlsx")
    tab_name = 'AllMTMOA'
    df = pd.read_excel(src_path, sheet_name=tab_name)
    destination_path = os.path.join(METADATA_DIR_PATH, f"mechanistic_target_and_mode_of_action.parquet.gzip")
    df.to_parquet(destination_path, compression='gzip')


def get_all_related_assay_infos(config):
    def convert_to_json_serializable(value):
        if isinstance(value, np.integer):
            return int(value)
        elif pd.isna(value):
            return None
        else:
            return value

    def save_distinct_values(column, values):
        out_file_path = os.path.join(METADATA_SUBSET_DIR_PATH, "assay_info_distinct_values", f"{column}.out")
        with open(out_file_path, 'w') as out_file:
            out_file.write('\n'.join(map(str, values)))

    os.makedirs(os.path.join(METADATA_SUBSET_DIR_PATH, "assay_info_distinct_values"), exist_ok=True)

    df_aeids = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids{config['file_format']}"))
    mechanistic_target_and_mode_of_action_df = pd.read_parquet(
        os.path.join(METADATA_DIR_PATH, f"mechanistic_target_and_mode_of_action{config['file_format']}"))
    assay_component_df = pd.read_parquet(os.path.join(METADATA_DIR_PATH, f"assay_component{config['file_format']}"))
    assay_component_endpoint_df = pd.read_parquet(
        os.path.join(METADATA_DIR_PATH, f"assay_component_endpoint{config['file_format']}"))

    assay_info_df = (
        assay_component_df.merge(assay_component_endpoint_df, on='acid')
        .merge(df_aeids[['aeid']], left_on='aeid', right_on='aeid')
        .merge(mechanistic_target_and_mode_of_action_df,
               left_on='assay_component_name', right_on='new_AssayEndpointName', how='left')
    )

    assay_info_distinct_values = {}
    with open(os.path.join(METADATA_SUBSET_DIR_PATH, "assay_info_distinct_values_counts.out"), 'w') as length_file:
        for column in assay_info_df.columns:
            distinct_values = assay_info_df[column].unique()

            # Convert ndarray to list and handle NaN values
            if isinstance(distinct_values[0], np.ndarray):
                distinct_values = [list(val) if isinstance(val, np.ndarray) else val for val in distinct_values]
            else:
                distinct_values = [convert_to_json_serializable(val) for val in distinct_values]

            assay_info_distinct_values[column] = distinct_values
            save_distinct_values(column, distinct_values)
            length_file.write(f"{column}: {len(distinct_values)}):\n")

    json_file_path = os.path.join(METADATA_SUBSET_DIR_PATH, "assay_info_distinct_values.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(assay_info_distinct_values, json_file)

    assay_info_df = assay_info_df.drop_duplicates(subset='aeid', keep='first')
    assay_info_df.to_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"assay_info{config['file_format']}"),
                             compression='gzip')

    return assay_info_df, assay_info_distinct_values