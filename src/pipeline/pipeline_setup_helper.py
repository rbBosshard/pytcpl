import os
import numpy as np
import pandas as pd
import json

from src.pipeline.pipeline_constants import METADATA_DIR_PATH, AEIDS_LIST_PATH, METADATA_SUBSET_DIR_PATH, FILE_FORMAT
from src.pipeline.pipeline_helper import query_db

pd.set_option('mode.chained_assignment', None)


def save_aeids(config, df):
    """
    Save AEIDs (Assay Endpoint IDs) to file for distributed processing.
    """
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
    """
    Generate a list of AEIDs (Assay Endpoint IDs) to process in parallel.
    """
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
    return df


def keep_viability_assay_endpoints_together(config, df):
    """
    Keep viability assay endpoints together with their counterparts.
    """
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


def omit_assays_from_ice_curation(df):
    """
    The cHTS data set in ICE omits the following assay endpoints:
    All Tanguay zebrafish assay endpoints
    All Tox21 assay endpoints comprising channel readouts. The Tox21 assays processed by the tcpl algorithm are analyzed as channel 1 and 2 ["ch1" and "ch2" at the end of the assay endpoint names], and their ratio. ICE cHTS provides only the ratio endpoint and not the raw channel outputs, despite their each being an assay endpoint in invitrodb.
    All Attagene assay endpoints analyzed in the down direction. These assays are transcriptional activation assays, which are only intended to be interpreted as increasing signal (up direction).
    All background readout assays as identified by assay endpoints having the term "background measurement" the "intended_target_family" field in invitrobd and/or the term "background control" in the "assay_function_type" field in invitrodb.
    """
    df = df[~(df['organism'] == "zebrafish")]
    df = df[~df['assay_component_endpoint_name'].str.endswith(('ch1', 'ch2'))]
    df = df[~((df['assay_component_endpoint_name'].str.startswith('ATG_')) & (df['signal_direction'] == 'loss'))]
    df = df[~((df['intended_target_family'] == 'background measurement') | (df['assay_function_type'] == 'background control'))]
    return df


def subset_for_candidate_assay_endpoints():
    """
    Subset for candidate assay endpoints.
    """
    os.makedirs(METADATA_SUBSET_DIR_PATH, exist_ok=True)
    dest_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"candidate_assay_endpoints.parquet.gzip")
    df1 = get_count_and_hit_ratio()
    df2 = filter_on_assay_format_type()  # Consider only cell-based assays
    df = df1.merge(df2, on="aeid", how="inner")
    df = omit_assays_from_ice_curation(df)
    df.to_parquet(dest_path, compression='gzip')
    return df


def filter_on_assay_format_type():
    """
    Filter assay endpoints on assay format type.
    """
    query = f"SELECT * " \
            f"FROM assay_component_endpoint AS ace " \
            f"INNER JOIN assay_component AS ac ON ace.acid = ac.acid " \
            f"INNER JOIN assay AS a ON ac.aid = a.AID " \
            f"WHERE assay_format_type IN ('cell-based') " \
            # f"AND assay_function_type NOT IN ('background control') " \
            # f"AND ace.analysis_direction = 'positive' AND signal_direction='gain' " \

    df = query_db(query)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    return df


# Estimate on hit ratio
def get_count_and_hit_ratio():
    """
    Get count and hit ratio for assay endpoints.
    """
    query = f"SELECT aeid, " \
            f"COUNT(*) as count, " \
            f"SUM(hitc >= 0.1) AS hitc_1_count, " \
            f"SUM(hitc < 0.1) AS hitc_0_count, " \
            f"SUM(hitc >= 0.1) / COUNT(*) AS ratio " \
            f"FROM mc5 " \
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
    tables = ["assay_source", "assay", "assay_component", "assay_component_endpoint",
              "assay_component_endpoint_descriptions",
              "sample", "chemical", "cytotox",
              "mc4_aeid", "mc5_aeid", "mc6_aeid",
              "mc4_methods", "mc5_methods", "mc6_methods"]

    for table in tables:
        df = query_db(f'SELECT * FROM {table}')
        destination_path = os.path.join(METADATA_DIR_PATH, f"{table}.parquet.gzip")
        df.to_parquet(destination_path, compression='gzip')

        if table == "assay_source":
            destination_path = os.path.join(METADATA_DIR_PATH, f"{table}.tex")
            df[['assay_source_name', 'assay_source_long_name']].to_latex(destination_path, index=False,formatters={"name": str.upper},float_format="{:.1f}".format)



def get_mechanistic_target_and_mode_of_action_annotations_from_ice():
    """
    Get mechanistic target and mode of action annotations from ICE.
    """
    # Download link: "https://ice.ntp.niehs.nih.gov/downloads/MOA/cHTSMT_ALL.xlsx"
    src_path = os.path.join(METADATA_DIR_PATH, f"cHTSMT_ALL.xlsx")
    tab_name = 'AllMTMOA'
    df = pd.read_excel(src_path, sheet_name=tab_name)
    destination_path = os.path.join(METADATA_DIR_PATH, f"mechanistic_target_and_mode_of_action.parquet.gzip")
    df.to_parquet(destination_path, compression='gzip')


def get_chemical_qc():
    """
    Get chemical QC data from ICE.
    """
    # Download link: "https://ice.ntp.niehs.nih.gov/downloads/MOA/ChemicalQC.xlsx"
    src_path = os.path.join(METADATA_DIR_PATH, f"ChemicalQC.xlsx")
    tab_name = 'cHTS Chemical QC DATA'
    df = pd.read_excel(src_path, sheet_name=tab_name)
    destination_path = os.path.join(METADATA_DIR_PATH, f"chemical_qc.parquet.gzip")
    df.to_parquet(destination_path, compression='gzip')
    compounds_qc_omit = df[df['NICEATM_qc_summary_call'] == 'QC_OMIT']
    compounds_qc_omit_df = pd.DataFrame({'dsstox_substance_id': compounds_qc_omit['dtxsid']})
    destination_path = os.path.join(METADATA_DIR_PATH, f"compounds_qc_omit.parquet.gzip")
    compounds_qc_omit_df.to_parquet(destination_path, compression='gzip')


def get_all_related_assay_infos(config, df):
    """
    Get all related assay info from ICE.
    """
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

    mechanistic_target_and_mode_of_action_df = pd.read_parquet(
        os.path.join(METADATA_DIR_PATH, f"mechanistic_target_and_mode_of_action{config['file_format']}"))

    assay_info_df = (
        df.merge(mechanistic_target_and_mode_of_action_df,
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
            length_file.write(f"{column}: {len(distinct_values)}\n")

    json_file_path = os.path.join(METADATA_SUBSET_DIR_PATH, "assay_info_distinct_values.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(assay_info_distinct_values, json_file)

    assay_info_df = assay_info_df.drop_duplicates(subset='aeid', keep='first')
    assay_info_df.to_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"assay_info{config['file_format']}"),
                             compression='gzip')

    return assay_info_df, assay_info_distinct_values


def adapt_viability_counterparts(target_df, viability_df):
    """
    Adapt viability counterparts.
    """
    viability_counterparts_name = {}
    viability_counterparts_aeid = {}
    for _, row in target_df.iterrows():
        name = row['assay_component_endpoint_name']
        aeid = row['aeid']
        aid = row['aid']
        counterpart_viability_assay_endpoint = viability_df[viability_df['aid'] == aid]
        if not counterpart_viability_assay_endpoint.empty:
            viability_counterparts_name[name] = str(counterpart_viability_assay_endpoint['assay_component_endpoint_name'].iloc[0])
            viability_counterparts_aeid[aeid] = str(counterpart_viability_assay_endpoint['aeid'].iloc[0])

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
    """
    Split assays.
    """
    viability_assay_endpoints = df[df['assay_component_endpoint_name'].str.endswith('viability')]
    target_assay_endpoints = df[~df['assay_component_endpoint_name'].str.endswith('viability')]
    return target_assay_endpoints, viability_assay_endpoints


def filter_on_count_and_hitcall_ratio(config, target_assay_endpoints):
    assays_to_drop = set()

    for _, row in target_assay_endpoints.iterrows():
        count = row['count']
        ratio = row['ratio']
        name = row['assay_component_endpoint_name']
        if count < config['threshold_subsetting_aeids_on_count_compounds_tested'] or ratio < config['threshold_subsetting_aeids_on_hit_ratio']:
            assays_to_drop.add(name)

    target_assay_endpoints = target_assay_endpoints.drop(target_assay_endpoints[target_assay_endpoints['assay_component_endpoint_name'].isin(assays_to_drop)].index)
    return target_assay_endpoints


def handle_viability_assays(config, df):
    """
    Handle viability assays for cytotoxicity correction.
    """
    is_burst_assay_endpoint = df['burst_assay'] == 1
    burst_assay_endpoints = df[is_burst_assay_endpoint]
    aeids_burst_assay_endpoints = df[is_burst_assay_endpoint]['aeid']
    is_viability_assay_endpoint = df['assay_function_type'].str.endswith('viability')
    viability_assay_endpoints = df[is_viability_assay_endpoint]

    target_assay_endpoints = df[~(is_burst_assay_endpoint | is_viability_assay_endpoint)]
    target_assay_endpoints = filter_on_count_and_hitcall_ratio(config, target_assay_endpoints)
    aeids_target_assays = target_assay_endpoints['aeid']

    viability_assay_endpoints = adapt_viability_counterparts(target_assay_endpoints, viability_assay_endpoints)
    aeids_target_viability_assays = viability_assay_endpoints['aeid']
    df = pd.concat([burst_assay_endpoints, target_assay_endpoints, viability_assay_endpoints], ignore_index=True)
    df = df.drop_duplicates().reset_index()

    destination_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_burst_assay_endpoints{FILE_FORMAT}")
    export_df = pd.DataFrame({'aeid': aeids_burst_assay_endpoints})
    export_df.to_parquet(destination_path, compression='gzip')
    destination_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_burst_assay_endpoints.csv")
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
