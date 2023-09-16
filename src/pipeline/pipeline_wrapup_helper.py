import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.pipeline.models.helper import get_mad
from src.pipeline.pipeline_helper import check_db, db_append, CONFIG
from src.pipeline.pipeline_constants import METADATA_SUBSET_DIR_PATH, OUTPUT_DIR_PATH, \
    CUTOFF_DIR_PATH, AEIDS_LIST_PATH, OUTPUT_COMPOUNDS_DIR_PATH, FILE_FORMAT, METADATA_DIR_PATH, DATA_DIR_PATH


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
    aeids_sorted = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_sorted{FILE_FORMAT}"))
    aeids_target_assays = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_target_assays{FILE_FORMAT}"))
    output_paths = [(aeid, os.path.join(OUTPUT_DIR_PATH, f"{aeid}{FILE_FORMAT}")) for aeid in aeids_sorted['aeid']]
    cutoff_paths = [os.path.join(CUTOFF_DIR_PATH, f"{aeid}{FILE_FORMAT}") for aeid in aeids_sorted['aeid']]
    json_file_path = os.path.join(METADATA_SUBSET_DIR_PATH, "viability_counterparts_aeid.json")
    with open(json_file_path, 'r') as json_file:
        viability_counterparts_aeid = json.load(json_file)

    assay_info_df = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"assay_info{FILE_FORMAT}"))

    dfs = {}
    for aeid, file_path in output_paths:
        df = pd.read_parquet(file_path)
        df["hitcall_c"] = df["hitcall"]
        df.loc[:, 'cytotox_flag'] = None
        df.loc[:, 'omit_flag'] = 0

        # Data curation: cHTS data in ICE have been curated to integrate concentration-response curve fit information and chemical QC data to bolster confidence in hit calls.
        # https://ice.ntp.niehs.nih.gov/DATASETDESCRIPTION?section=cHTS
        assay_and_chemical_based_ice_curation(aeid, assay_info_df, df)

        # Correct for cytotoxicity in target assasy with corresponding viability assay
        if aeid in aeids_target_assays:
            df = correct_for_cytotoxicity(str(aeid), df, viability_counterparts_aeid)

        dfs[aeid] = df

    df_all = pd.concat(dfs.values(), axis=0)
    cutoff_all = pd.concat([pd.read_parquet(file) for file in cutoff_paths])
    return df_all, cutoff_all


def assay_and_chemical_based_ice_curation(aeid, assay_info_df, df):
    # Curve/assay-based curation:
    # For NovaScreen (NVS) cell-free biochemical assays, any active calls with less than 50% efficacy.
    condition_assay = assay_info_df['assay_component_endpoint_name'].str.startswith('NVS_') & (
                assay_info_df['cell_format'] == 'cell_free')
    if aeid in assay_info_df[condition_assay]['aeid']:
        condition_omit_flag = df['top'] < 0.5  # NovaScreen (NVS) cell-free biochemical assays normalized as percent_activity
        df.loc[condition_omit_flag, 'omit_flag'] = 1
    # For NCATS Tox21 assays, any active calls where only the lowest concentration tested exceeded the assay activity cutoff threshold and the best-fit curve was a gain-loss model.
    # Note: NCATS Tox21 assays not found
    # Any down-direction (i.e., inhibition, antagonism, loss-of-signal, etc.) active call where the best-fit curve was a gain-loss model.
    if aeid in assay_info_df[assay_info_df['signal_direction'] == 'loss']['aeid']:
        condition_omit_flag = df['best_aic_model'].isin(
            ['gnls', 'sigmoid'])  # gnls and sigmoid both model loss of activity for increasing concentrations
        df.loc[condition_omit_flag, 'omit_flag'] = 2
    concs = df['conc'].apply(json.loads)
    # Any active call where the best-fit curve was a gain-loss model and AC50 that was extrapolated below the testing concentration range.
    min_concs = list(map(min, concs))
    condition_omit_flag = df['best_aic_model'].isin(['gnls', 'sigmoid']) & (df['ac50'] < min_concs)
    df.loc[condition_omit_flag, 'omit_flag'] = 3
    # Any active concentration-response curve where the AC50 was extrapolated to a concentration above the tested concentration range.
    max_concs = list(map(max, concs))
    condition_omit_flag = df['ac50'] > max_concs
    df.loc[condition_omit_flag, 'omit_flag'] = 4
    # Any active call with flags 11 and 16 from the tcpl pipeline, which suggest marginal efficacy and likely overfitting.
    # Note: fitc flags 11 and 16 not availbale in new (py)tcpl version. Map new quality warning flags

    # ICE omits for the cHTS data set entire assay endpoints (already done in pipeline setup by subset aeids):

    # Chemical QC-based curation:
    destination_path = os.path.join(METADATA_DIR_PATH, f"compounds_qc_omit.parquet.gzip")
    compounds_qc_omit_df = pd.read_parquet(destination_path)
    condition_omit_flag = df['dsstox_substance_id'].isin(compounds_qc_omit_df['dsstox_substance_id'])
    df.loc[condition_omit_flag, 'omit_flag'] = 5


def correct_for_cytotoxicity(aeid, df, viability_counterparts_aeid):
    # For cell based assays, try finding matching viability/burst/background assays and correct hitc where
    # AC50 in the former > than AC50 in the latter (we can adjust boundaries around that).
    if aeid in viability_counterparts_aeid.values():
        df.loc[:, 'cytotox_flag'] = 0

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
                    df.at[index, 'cytotox_flag'] = 20
                else:
                    df.at[index, 'cytotox_flag'] = 30

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


def cytotoxicity_curation_with_burst_assays(config, df_all):
    # Some names have changed in the new version but is equivalent, e.g. cytotox_median_um = cyto_pt_um
    cytotox = compute_cytotoxicity_from_burst_assays(config, df_all)
    missing_vals = cytotox["cyto_pt_um"].isna()
    if missing_vals.sum() == 0:
        print("No incomplete cytotoxicity data")
    else:
        cytotox_incomplete = cytotox[missing_vals]
        print(f"Incomplete cytotoxicity data; nrows={cytotox_incomplete.shape[0]}")
        cytotox.loc[missing_vals, "cyto_pt_um"] = cytotox["cyto_pt_um"].median()

    # Plot overview of cytotoxicity data
    plot_overview_of_cytotoxicity_data(cytotox)

    # SPlit active and inactive cases
    df_all_act = df_all[df_all["hitcall_c"] > 0]
    df_all_inact = df_all[df_all["hitcall_c"] == 0]

    # Calculate acc_um and join cytotox to specific tox data from mc5
    df_all_act.loc[:, "acc_um"] = df_all_act["acc"]  # acc is saved as non-log concentration
    df_all_act = df_all_act.merge(cytotox, on="dsstox_substance_id", how="left")

    # Flag different cases
    flag_cytotoxicity_cases_based_on_burst_assays(df_all_act)

    # Print summary of flags
    print_summary_of_cytotox_flags(df_all_act)

    # Determination of binary cytotoxicity: cytotoxic/not cytotoxic, and inconclusive
    df_all_act["ctx_acc"] = "inconclusive"
    df_all_act.loc[(df_all_act["cytotox_flag"] >= 10) & (df_all_act["cytotox_flag"] != 50), "ctx_acc"] = "cytotoxic"
    df_all_act.loc[df_all_act["cytotox_flag"] <= 2, "ctx_acc"] = "not cytotoxic"

    # Calculate hitc_acc
    df_all_act["hitcall_c"] = np.where(df_all_act["ctx_acc"] == "cytotoxic", 0.0, df_all_act["hitcall_c"])

    # Count
    count_hitc = pd.DataFrame({'hitcall': ["inactive", "active"], 'count': [(df_all["hitcall"] == 0).sum(), (df_all["hitcall"] > 0).sum()]})
    count_hitacc = df_all_act["ctx_acc"].value_counts(dropna=False).reset_index()
    count_hitacc.columns = ["ctx_acc", "count"]
    count_cytotox_flag = df_all_act["cytotox_flag"].value_counts().reset_index()
    count_cytotox_flag.columns = ["cytotox_flag", "count"]
    count_cytotox_flag["col"] = "cyan"
    count_cytotox_flag.loc[count_cytotox_flag["cytotox_flag"] >= 10, "col"] = "magenta"
    count_cytotox_flag.loc[count_cytotox_flag["cytotox_flag"] == 50, "col"] = "grey"

    # Plot an overview of the flagging and hit calls
    plot_overview_cytotoxicity_flagging_and_hitcalls(count_cytotox_flag, count_hitacc, count_hitc)

    # Combine active and inactive data frames
    df_all_c = merge_active_and_inactive_df(df_all_act, df_all_inact)

    return df_all_c


def merge_active_and_inactive_df(df, df_all_inact):
    # Get common columns between df and df_all_inact
    common_cols = list(set(df.columns) & set(df_all_inact.columns))
    df_common = df[common_cols]
    df_all_inact_common = df_all_inact[common_cols]
    # Concatenate the two data frames with only the common columns and save
    df_all_c = pd.concat([df_common, df_all_inact_common], ignore_index=True)
    df_all_c['hitcall_c'] = df_all_c['hitcall_c'].astype(float)
    # folder_path = os.path.join(DATA_DIR_PATH, "merged", "output_cytotox_flagged")
    # os.makedirs(folder_path, exist_ok=True)
    # file_path = os.path.join(folder_path, f"{0}{FILE_FORMAT}")
    # df_all_c.to_parquet(file_path, compression='gzip')
    return df_all_c


def plot_overview_cytotoxicity_flagging_and_hitcalls(count_cytotox_flag, count_hitacc, count_hitc):
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    axes = axes.flatten()
    titles = ["Hit calls in relevant target subset of assay endpoints", "Cytotox. flags for positive hit calls",
              "Positive hit calls, filtered for cytotox.", "Hit calls after cytotox filtering"]
    x_labels = ["Hitcall", "Cytotox. flag", "Hitcall, ct filtered", "Hitcall"]
    y_labels = ["Count"] * 4
    # Define custom qualitative colors
    color_group2 = "#FFCC66"  # Color for cytotox_flag 0-2
    color_group1 = "#78AB46"  # Color for cytotox_flag 10-12
    color_group3 = "grey"  # Color for cytotox_flag 50
    # Assign colors to different flag groups
    flag_colors = np.where(count_cytotox_flag["cytotox_flag"].isin([0, 1, 2]), color_group2,
                           np.where(count_cytotox_flag["cytotox_flag"].isin([10, 11, 12]), color_group1, color_group3))
    colors_list = [["blue", "red"], flag_colors, ["blue", "red", "grey"], ["blue", "red"]]
    categories_list = [count_hitc["hitcall"].unique(), count_cytotox_flag["cytotox_flag"].unique(),
                       count_hitacc["ctx_acc"].unique(), count_hitc["hitcall"].unique()]
    categories_list = [[str(value) for value in inner_list] for inner_list in categories_list]
    labels_list = [None, ["inconclusive", "cytotoxic", "not cytotoxic"], None, None]
    values_list = [count_hitc["count"], count_cytotox_flag["count"], count_hitacc["count"],
                   [  # inactive with cytotox filtering
                       count_hitc[(count_hitc['hitcall'] == "inactive")]['count'].iloc[0] -
                       count_hitacc[(count_hitacc['ctx_acc'] == "cytotoxic")]['count'].iloc[0],
                       # active with cytotox filtering
                       count_hitacc[(count_hitacc['ctx_acc'] == "not cytotoxic")]['count'].iloc[0]]]
    for i, (ax, values, categories, colors, title, x_label, y_label, labels) in enumerate(
            zip(axes, values_list, categories_list, colors_list, titles, x_labels, y_labels, labels_list)):
        ax.barh(categories, values, color=colors, label=labels)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_yticks(categories)
        # ax.set_yticks([])
        if labels is not None:
            ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(METADATA_SUBSET_DIR_PATH, "counts_hitc_cytotox_flags.pdf"))
    plt.close()


def print_summary_of_cytotox_flags(df):
    flag_counts = df["cytotox_flag"].value_counts(dropna=False)
    print("High conf. of cytotoxicity:", flag_counts.get(10, 0))
    print("Some conf. of cytotoxicity:", flag_counts.get(11, 0))
    print("Possible cytotoxicity:", flag_counts.get(12, 0))
    print("High conf. of no cytotoxicity:", flag_counts.get(0, 0))
    print("Some conf. of no cytotoxicity:", flag_counts.get(1, 0))
    print("Unlikely cytotoxicity:", flag_counts.get(2, 0))
    print("No data:", flag_counts.get(50, 0))
    return flag_counts


def flag_cytotoxicity_cases_based_on_burst_assays(df):
    # Flag = 10: High confidence of cytotoxicity rather than specific toxicity
    upperlim = df["cyto_pt_um"] + (df["cyto_pt_um"] - df["lower_bnd_um"])
    df.loc[(df["acc_um"] >= upperlim) & (df["nhit"] >= 5), "cytotox_flag"] = 10
    # Flag = 11: Some confidence of cytotoxicity
    midlim = df["cyto_pt_um"] + 0.5 * (df["cyto_pt_um"] - df["lower_bnd_um"])
    df.loc[(df["acc_um"] >= midlim) & (df["acc_um"] < upperlim) & (df["nhit"] >= 3), "cytotox_flag"] = 11
    # Flag = 0: High confidence of no cytotoxicity
    df.loc[(df["acc_um"] < df["lower_bnd_um"]) & (df["ntst"] >= 5), "cytotox_flag"] = 0
    # Flag = 1: Some confidence of no cytotoxicity
    lowlim = df["cyto_pt_um"] - 0.5 * (df["cyto_pt_um"] - df["lower_bnd_um"])
    df.loc[(df["acc_um"] < lowlim) & (df["acc_um"] >= df["lower_bnd_um"]) & (df["ntst"] >= 3), "cytotox_flag"] = 1
    # Flag = 50: No assays conducted; cannot determine cytotoxicity
    df.loc[df["ntst"] == 0, "cytotox_flag"] = 50
    df.loc[df["cyto_pt_um"].isna(), "cytotox_flag"] = 50
    # Flag = 12: Tendentially cytotoxic but unsure
    df.loc[(df["acc_um"] >= df["cyto_pt_um"]) & df["cytotox_flag"].isna(), "cytotox_flag"] = 12
    # Flag = 2: Tendentially not cytotoxic but unsure
    df.loc[(df["acc_um"] <= df["cyto_pt_um"]) & df["cytotox_flag"].isna(), "cytotox_flag"] = 2
    # Flag = 50: all remaining cases. Todo: Uncomment if this is desired
    # df.loc[df["cytotox_flag"].isna(), "cytotox_flag"] = 50


def plot_overview_of_cytotoxicity_data(cytotox):
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    axes = axes.flatten()
    titles = ["cytotox_median_raw", "cytotox_mad", "cytotox_lower_bound_um", "ntested", "cytotox_median_um", "nhit"]
    x_labels = titles
    y_labels = ["Density"] * 6
    x_ranges = [None, (-0.1, 2), (-1, 120), (-1, 100), (-1, 120), (-1, 100)]
    y_ranges = [None, None, (0, 0.1), None, (0, 0.02), None]
    bins_list = [50, 100, 500, 50, 100, 50]
    legend_notes = ["(all values shown)",
                    "(values > 1 not shown)",
                    "(values == 1000 not shown)",
                    "(all values shown)",
                    "(values > 120 not shown)",
                    "(all values shown)"]
    data = [cytotox["cyto_pt"], cytotox["mad"], cytotox["lower_bnd_um"], cytotox["ntst"], cytotox["cyto_pt_um"],
            cytotox["nhit"]]
    for i, (ax, data, bins, title, x_label, y_label, legend_note, x_range, y_range) in enumerate(
            zip(axes, data, bins_list, titles, x_labels, y_labels, legend_notes, x_ranges, y_ranges)):
        ax.hist(data, bins=bins)  # label=legend_note
        ax.set_title(legend_note)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_range is not None:
            ax.set_xlim(x_range)
        if y_range is not None:
            ax.set_xlim(y_range)
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(METADATA_SUBSET_DIR_PATH, "cytotox_overview_alldata.pdf"))
    plt.close()
