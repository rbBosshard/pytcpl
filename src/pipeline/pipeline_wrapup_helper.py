import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from src.pipeline.models.helper import get_mad
from src.pipeline.pipeline_helper import check_db, db_append, CONFIG
from src.pipeline.pipeline_constants import METADATA_SUBSET_DIR_PATH, OUTPUT_DIR_PATH, \
    CUTOFF_DIR_PATH, AEIDS_LIST_PATH, OUTPUT_COMPOUNDS_DIR_PATH, FILE_FORMAT, METADATA_DIR_PATH


def remove_files_not_matching_to_aeid_list(delete=False):
    directories = [OUTPUT_DIR_PATH, CUTOFF_DIR_PATH]

    df = pd.read_csv(os.path.join(METADATA_SUBSET_DIR_PATH, 'aeids.csv'))
    ids = set(df['aeid'])

    ids_files = []
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith('.parquet.gzip'):
                id = filename.replace('.parquet.gzip', '')
                ids_files.append(id)
                if id not in ids:
                    if delete:  # deletes files not contained in aeid list
                        file_path = os.path.join(directory, filename)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Deleted: {file_path}")
                        else:
                            print(f"Warning: File not found: {file_path} (aeid: {id})")

    for id in ids:
        if id not in ids_files:
            print(f"{id}")


def save_merged(df_all, cutoff_all):
    # For writing to DB ensure it holds that in config/config.yaml: enable_writing_db: 1
    # check_db()
    print("Takes approx. 10 minutes depending on CPUs")
    db_append(cutoff_all, 'cutoff')
    db_append(df_all, 'output')
    return df_all, cutoff_all


def groupb_by_compounds(config, df_all):
    compounds = df_all['dsstox_substance_id'].dropna().unique()
    compounds_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"compounds_tested{config['file_format']}")
    compounds_df = pd.DataFrame(compounds, columns=['dsstox_substance_id'])
    compounds_df.to_parquet(compounds_path, compression='gzip')
    compounds_path = os.path.join(METADATA_SUBSET_DIR_PATH, f"compounds_tested.csv")
    compounds_df.to_csv(compounds_path, index=False)

    num_compounds = len(compounds)
    print(f"Number of compounds tested: {num_compounds}")

    if config['enable_output_compounds']:
        get_compound_results(df_all, num_compounds, compounds)


def get_compound_results(df_all, num_compounds, unique_compounds):
    def process_compound(i, compound):
        print(f"{i + 1}/{num_compounds}: {compound}")
        compound_df = df_all[df_all['dsstox_substance_id'] == compound]
        output_file = os.path.join(OUTPUT_COMPOUNDS_DIR_PATH, f'{compound}.parquet.gzip')
        compound_df.to_parquet(output_file, compression='gzip')

    os.makedirs(OUTPUT_COMPOUNDS_DIR_PATH, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(os.cpu_count() * 2 - 1, 1)) as executor:
        for i, compound in enumerate(unique_compounds):
            executor.submit(process_compound, i, compound)


def ice_curation_and_cytotoxicity_filtering_with_viability_assays():
    aeids_sorted = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_sorted{FILE_FORMAT}"))[:20]
    aeids_target_assays = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_target_assays{FILE_FORMAT}"))
    output_paths = [(aeid, os.path.join(OUTPUT_DIR_PATH, f"{aeid}{FILE_FORMAT}")) for aeid in aeids_sorted['aeid']]
    cutoff_paths = [os.path.join(CUTOFF_DIR_PATH, f"{aeid}{FILE_FORMAT}") for aeid in aeids_sorted['aeid']]
    json_file_path = os.path.join(METADATA_SUBSET_DIR_PATH, "viability_counterparts_aeid.json")
    with open(json_file_path, 'r') as json_file:
        viability_counterparts_aeid = json.load(json_file)

    assay_info_df = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"assay_info{FILE_FORMAT}"))

    dfs = []
    for aeid, file_path in output_paths:
        df = pd.read_parquet(file_path)
        df["hitcall_c"] = df["hitcall"]
        df.loc[:, 'cytotoxicity_class'] = None
        df.loc[:, 'cytotox_flag'] = None
        df.loc[:, 'omit_flag'] = 0

        # Data curation:
        # cHTS data in ICE have been curated to integrate concentration-response curve fit information and chemical QC data to bolster confidence in hit calls.
        # https://ice.ntp.niehs.nih.gov/DATASETDESCRIPTION?section=cHTS

        # Curve/assay-based curation:
        # For NovaScreen (NVS) cell-free biochemical assays, any active calls with less than 50% efficacy.
        condition_assay = assay_info_df['assay_component_endpoint_name'].str.startswith('NVS_') & (df['cell_format'] == 'cell_free')
        if aeid in assay_info_df[condition_assay]['aeid']:
            condition_omit_flag = df['top'] < 0.5  # NovaScreen (NVS) cell-free biochemical assays normalized as percent_activity
            df.loc[df[condition_omit_flag], 'omit_flag'] = 1

        # For NCATS Tox21 assays, any active calls where only the lowest concentration tested exceeded the assay activity cutoff threshold and the best-fit curve was a gain-loss model.
        # Note: NCATS Tox21 assays not found

        # Any down-direction (i.e., inhibition, antagonism, loss-of-signal, etc.) active call where the best-fit curve was a gain-loss model.
        if aeid in assay_info_df[assay_info_df['signal_direction'] == 'loss']['aeid']:
            condition_omit_flag = df['best_aic_model'] in ['gnls', 'sigmoid']  # gnls and sigmoid both model loss of activity for increasing concentrations
            df.loc[df[condition_omit_flag], 'omit_flag'] = 2

        # Any active call where the best-fit curve was a gain-loss model and AC50 that was extrapolated below the testing concentration range.
        condition_omit_flag = (df['best_aic_model'] in ['gnls', 'sigmoid']) & (df['ac50'] < np.min(df['conc']))
        df.loc[df[condition_omit_flag], 'omit_flag'] = 3

        # Any active concentration-response curve where the AC50 was extrapolated to a concentration above the tested concentration range.
        condition_omit_flag = df['ac50'] > np.max(df['conc'])
        df.loc[df[condition_omit_flag], 'omit_flag'] = 4

        # Any active call with flags 11 and 16 from the tcpl pipeline, which suggest marginal efficacy and likely overfitting.
        # Note: fitc flags 11 and 16 not availbale in new (py)tcpl version. Map new quality warning flags

        # ICE omits for the cHTS data set entire assay endpoints (already done in pipeline setup by subset aeids):

        # Chemical QC-based curation:
        destination_path = os.path.join(METADATA_DIR_PATH, f"compounds_qc_omit.parquet.gzip")
        compounds_qc_omit_df = pd.read_parquet(destination_path)
        condition_omit_flag = df['aeid'].isin(compounds_qc_omit_df['aeid'])
        df.loc[df[condition_omit_flag], 'omit_flag'] = 5

        # Correct for cytotoxicity in target assasy with corresponding viability assay
        if aeid in aeids_target_assays:
            df = correct_for_cytotoxicity(str(aeid), df, viability_counterparts_aeid)

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    cutoff_all = pd.concat([pd.read_parquet(file) for file in cutoff_paths])
    return df_all, cutoff_all


def correct_for_cytotoxicity(aeid, df, viability_counterparts_aeid):
    # For cell based assays, try finding matching viability/burst/background assays and correct hitc where
    # AC50 in the former > than AC50 in the latter (we can adjust boundaries around that).
    if aeid in viability_counterparts_aeid.values():
        df.loc[:, 'cytotoxicity_class'] = -1

    if aeid in viability_counterparts_aeid:
        viability_assay_endpoint_aeid = viability_counterparts_aeid[aeid]
        viability_assay_endpoint_path = os.path.join(OUTPUT_DIR_PATH, f"{viability_assay_endpoint_aeid}{FILE_FORMAT}")
        viability_assay_endpoint = pd.read_parquet(viability_assay_endpoint_path)

        for index, row in df.iterrows():
            compound = row['dsstox_substance_id']
            ac50 = row['ac50']

            # Find the corresponding compounds in viability_assay_endpoint
            matching_row = viability_assay_endpoint[viability_assay_endpoint['dsstox_substance_id'] == compound]

            if not matching_row.empty:
                ac50_viability = matching_row['ac50'].iloc[0]
                hitcall_viability = matching_row['hitcall'].iloc[0]
                best_aic_model_viability = matching_row['best_aic_model'].iloc[0]

                # Compare AC50 values and update 'hitcall' in target assay endpoint if needed
                contending_cytotoxicity = ac50_viability < ac50 \
                                          and hitcall_viability > 0.9 \
                                          and best_aic_model_viability not in ['gnls', 'sigmoid']

                if contending_cytotoxicity:
                    df.at[index, 'hitcall_c'] = 0
                    df.at[index, 'cytotoxicity_class'] = 2
                else:
                    df.at[index, 'cytotoxicity_class'] = 1

    target_assay_endpoint_path = os.path.join(OUTPUT_DIR_PATH, f"{aeid}{FILE_FORMAT}")
    df.to_parquet(target_assay_endpoint_path, compression='gzip')

    return df


def compute_cytotoxicity_from_burst_assays(config, df_all):
    path = os.path.join(METADATA_DIR_PATH, f"assay_component_endpoint{FILE_FORMAT}")
    df = pd.read_parquet(path)
    burst_assays_aeids = df[df['burst_assay'] == 1]['aeid']
    threshold_cytotox_min_tested = config['threshold_cytotox_min_test']
    # Todo: Check why min_test not used is intended, here burst_assays['burstpct'] > 0.05 is used
    min_test = int(0.8 * len(burst_assays_aeids)) if threshold_cytotox_min_tested <= 1 else threshold_cytotox_min_tested

    burst_assays = df_all[df_all['aeid'].isin(burst_assays_aeids)]
    # burst_assays = burst_assays[~burst_assays['best_aic_model'].isin(['gnls', 'sigmoid'])]
    hitc_num = config['threshold_cytotox_hitc_num']

    def compute_metrics(x):
        med = np.median(np.log10(x.loc[x['hitcall'] >= hitc_num, 'ac50']))
        mad = get_mad(np.log10(x.loc[x['hitcall'] >= hitc_num, 'ac50']))
        ntst = len(x)
        nhit = np.sum(x['hitcall'] >= hitc_num)
        burstpct = nhit / ntst

        result = {"med": med, "mad": mad, "ntst": ntst, "nhit": nhit, "burstpct": burstpct}
        return pd.Series(result, name="metrics")

    burst_assays = burst_assays.groupby(['dsstox_substance_id']).apply(compute_metrics).reset_index()

    burst_assays['used_in_global_mad_calc'] = burst_assays['burstpct'] > 0.05
    gb_mad = np.median(burst_assays[burst_assays['used_in_global_mad_calc']]['mad'])
    burst_assays['global_mad'] = gb_mad
    burst_assays['cyto_pt'] = burst_assays['med']
    burst_assays.loc[burst_assays['burstpct'] < 0.05, 'cyto_pt'] = config['threshold_cytotox_default_pt']
    burst_assays['cyto_pt_um'] = 10 ** burst_assays['cyto_pt']
    burst_assays['lower_bnd_um'] = 10 ** (burst_assays['cyto_pt'] - 3 * gb_mad)
    burst_assays = burst_assays.drop(columns=['burstpct'])
    path = os.path.join(METADATA_DIR_PATH, f"cytotox_{FILE_FORMAT}")
    burst_assays.to_parquet(path, compression='gzip')
    path = os.path.join(METADATA_DIR_PATH, f"cytotox_.csv")
    burst_assays.to_csv(path)

    return burst_assays


def groupb_by_aeids(df_all):
    aeids = df_all['aeid'].dropna().unique()
    num_aeids = len(aeids)

    def process_compound(i, aeid):
        print(f"{i + 1}/{num_aeids}: {aeid}")
        aeid_df = df_all[df_all['aeid'] == aeid]
        output_file = os.path.join(OUTPUT_DIR_PATH, f'{aeid}{FILE_FORMAT}')
        aeid_df.to_parquet(output_file, compression='gzip')

    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(os.cpu_count() * 2 - 1, 1)) as executor:
        for i, aeid in enumerate(aeids):
            executor.submit(process_compound, i, aeids)
