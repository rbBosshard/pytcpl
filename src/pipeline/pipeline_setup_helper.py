import os
import numpy as np
import pandas as pd
import json

from src.pipeline.pipeline_constants import METADATA_DIR_PATH, AEIDS_LIST_PATH, METADATA_SUBSET_DIR_PATH
from src.pipeline.pipeline_helper import query_db

pd.set_option('mode.chained_assignment', None)


def save_aeids(config, df):
    aeids = df['aeid'].unique()
    aeids_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids{config['file_format']}")
    aeids_df = pd.DataFrame(aeids, columns=['aeid'])
    aeids_df.to_parquet(aeids_path, compression='gzip')
    aeids_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids.csv")
    aeids_df.to_csv(aeids_path, index=False)

    aeids_sorted = np.sort(aeids)
    aeids_sorted_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_sorted{config['file_format']}")
    aeids_sorted_df = pd.DataFrame(aeids_sorted, columns=['aeid'])
    aeids_sorted_df.to_parquet(aeids_sorted_path, compression='gzip')
    aeids_sorted_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_sorted.csv")
    aeids_sorted_df.to_csv(aeids_sorted_path, index=False)

    return aeids


def generate_balanced_aeid_list(config, df):
    df = df.sort_values('hitc_1_count', ascending=False).reset_index(drop=True)
    aeids = save_aeids(config, df)
    num_aeids = len(aeids)
    tasks_per_instance = (len(aeids) + config['instances_total'] - 1) // config['instances_total']
    distributed_tasks = distribute_aeids_to_instances(aeids, config['instances_total'])
    with open(AEIDS_LIST_PATH, "w") as file:
        for i, instance_tasks in enumerate(distributed_tasks):
            for task_id in instance_tasks[:tasks_per_instance]:
                file.write(str(task_id) + "\n")
    print(f"Instances total: {config['instances_total']}")
    print(f"Total num aeids to process: {num_aeids}")
    print(f"Num aeids per instance to process: {tasks_per_instance}")


def keep_viability_assay_endpoints_together(config, df):
    # Separate dataframes based on endpoint names
    ratio_df = df[
        df['assay_component_endpoint_name'].str.endswith("_ratio") & (
                    df['count'] > config['threshold_subsetting_aeids_on_count_compounds_tested']) & (
                df['ratio'] > config['threshold_subsetting_aeids_on_hit_ratio'])]
    viability_df = df[df['assay_component_endpoint_name'].str.endswith("_viability")]
    filtered_df = df[~df['assay_component_endpoint_name'].str.endswith(("_ratio", "_viability", "_ch1", "_ch2")) & (
            df['count'] > config['threshold_subsetting_aeids_on_count_compounds_tested']) & (
                                 df['ratio'] > config['threshold_subsetting_aeids_on_hit_ratio'])]
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


def sort_out_assay_endpoints_from_ice_curation(df):
    """
    The cHTS data set in ICE omits the following assay endpoints:
    All Tanguay zebrafish assay endpoints
    All Tox21 assay endpoints comprising channel readouts. The Tox21 assays processed by the tcpl algorithm are analyzed as channel 1 and 2 ["ch1" and "ch2" at the end of the assay endpoint names], and their ratio. ICE cHTS provides only the ratio endpoint and not the raw channel outputs, despite their each being an assay endpoint in invitrodb.
    All Attagene assay endpoints analyzed in the down direction. These assays are transcriptional activation assays, which are only intended to be interpreted as increasing signal (up direction).
    All background readout assays as identified by assay endpoints having the term "background measurement" the "intended_target_family" field in invitrobd and/or the term "background control" in the "assay_function_type" field in invitrodb.
    """
    df = df[~(df['organism'] == "zebrafish")]
    df = df[~df['assay_component_endpoint_name'].str.endswith(('ch1', 'ch2'))]
    df = df[~((df['assay_component_endpoint_name'].str.endswith('ATG_')) & (df['analysis_direction'] == 'negative'))]
    df = df[~((df['intended_target_family'] == 'background measurement') | (df['assay_function_type'] == 'background control'))]
    return df


def subset_for_candidate_assay_endpoints():
    os.makedirs(METADATA_SUBSET_DIR_PATH, exist_ok=True)
    dest_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"candidate_assay_endpoints.parquet.gzip")
    # if not os.path.exists(dest_path):
    df_1 = get_count_and_hit_ratio()
    df_2 = filter_on_assay_format_and_function_type()
    df = df_1.merge(df_2, on="aeid", how="inner")
    df = sort_out_assay_endpoints_from_ice_curation(df)
    df.to_parquet(dest_path, compression='gzip')
    # else:
    #     df = pd.read_parquet(dest_path)
    return df


def filter_on_assay_format_and_function_type():
    query = f"SELECT ace.aeid, " \
            f"ace.assay_component_endpoint_name, " \
            f"ace.burst_assay, " \
            f"ace.intended_target_family, " \
            f"ace.assay_function_type, " \
            f"ace.analysis_direction, " \
            f"a.assay_format_type, " \
            f"a.organism " \
            f"FROM invitrodb_v3o5.assay_component_endpoint AS ace " \
            f"INNER JOIN assay_component AS ac ON ace.acid = ac.acid " \
            f"INNER JOIN assay AS a ON ac.aid = a.AID " \
            f"WHERE assay_format_type IN ('cell-based') " \
            f"AND assay_function_type NOT IN ('background control') " \
            # f"AND ace.analysis_direction = 'positive' AND signal_direction='gain' " \

    df = query_db(query)
    return df


def get_count_and_hit_ratio():
    query = f"SELECT aeid, " \
            f"COUNT(*) as count, " \
            f"SUM(hitc = 1) AS hitc_1_count, " \
            f"SUM(hitc = 0) AS hitc_0_count, " \
            f"SUM(hitc = 1) / COUNT(*) AS ratio " \
            f"FROM invitrodb_v3o5.mc5 " \
            f"GROUP BY aeid;"
    df = query_db(query)
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
    tables = ["assay", "assay_component", "assay_component_endpoint",
              "assay_component_endpoint_descriptions",
              "sample", "chemical", "cytotox",
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
    return df


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
        with open(out_file_path, 'w', encoding='utf-8') as out_file:
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
            length_file.write(f"{column}: {len(distinct_values)}:\n")

    json_file_path = os.path.join(METADATA_SUBSET_DIR_PATH, "assay_info_distinct_values.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(assay_info_distinct_values, json_file)

    assay_info_df = assay_info_df.drop_duplicates(subset='aeid', keep='first')
    assay_info_df.to_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"assay_info{config['file_format']}"),
                             compression='gzip')

    return assay_info_df, assay_info_distinct_values
