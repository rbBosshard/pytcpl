import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import matplotlib.cm as cm
import seaborn as sns
import matplotlib
from scipy.stats import norm
matplotlib.use('Agg')


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
    """
    Save merged data frame and cutoff data frame to parquet files.
    """
    # For writing to DB ensure it holds that in config/config.yaml: enable_writing_db: 1
    # check_db()
    print("Takes approx. 10 minutes depending on CPUs")
    db_append(cutoff_all, 'cutoff', db=False)
    db_append(df_all, 'output', db=False)
    return df_all, cutoff_all


def groupb_by_compounds(config, df_all):
    """
    Group by compounds and save to parquet files.
    """
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
    """
    Get results for each compound and save to parquet files.
    """
    def process_compound(i, compound):
        print(f"{i + 1}/{num_compounds}: {compound}")
        compound_df = df_all[df_all['dsstox_substance_id'] == compound]
        output_file = os.path.join(OUTPUT_COMPOUNDS_DIR_PATH, f'{compound}.parquet.gzip')
        compound_df.to_parquet(output_file, compression='gzip')

    os.makedirs(OUTPUT_COMPOUNDS_DIR_PATH, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(os.cpu_count() * 2 - 1, 1)) as executor:
        for i, compound in enumerate(unique_compounds):
            executor.submit(process_compound, i, compound)


def ice_curation_and_cytotoxicity_handling(config):
    """
    Perform ICE curation and cytotoxicity handling.
    """
    aeids_sorted = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_sorted{FILE_FORMAT}"))
    aeids_target_assays = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"aeids_target_assays{FILE_FORMAT}"))
    output_paths = [(aeid, os.path.join(OUTPUT_DIR_PATH, f"{aeid}{FILE_FORMAT}"), os.path.join(CUTOFF_DIR_PATH, f"{aeid}{FILE_FORMAT}")) for aeid in aeids_sorted['aeid']]
    cutoff_paths = [os.path.join(CUTOFF_DIR_PATH, f"{aeid}{FILE_FORMAT}") for aeid in aeids_sorted['aeid']]
    json_file_path = os.path.join(METADATA_SUBSET_DIR_PATH, "viability_counterparts_aeid.json")
    with open(json_file_path, 'r') as json_file:
        viability_counterparts_aeid = json.load(json_file)

    assay_info_df = pd.read_parquet(os.path.join(METADATA_SUBSET_DIR_PATH, f"assay_info{FILE_FORMAT}"))

    def post_process(aeid, aeid_path, cutoff_path, assay_info_df, aeids_target_assays, viability_counterparts_aeid):
        df = pd.read_parquet(aeid_path)
        # cutoff = pd.read_parquet(cutoff_path)['cutoff']
        # Init new fields relevant for post-processing
        df.loc[:, 'omit_flag'] = "PASS"
        df["hitcall_c"] = df["hitcall"]
        df.loc[:, 'cytotox_ref'] = 'burst'
        df.loc[:, 'cytotox_prob'] = None
        df.loc[:, 'cytotox_acc'] = 1000
        df.loc[:, 'viability_aeid'] = None

        # Data curation: cHTS data in ICE have been curated to integrate concentration-response curve fit information and chemical QC data to bolster confidence in hit calls.
        # https://ice.ntp.niehs.nih.gov/DATASETDESCRIPTION?section=cHTS
        if config['enable_ICE_filtering']:
            ice_curation_adding_omit_flags(aeid, assay_info_df, df)

        # Cytotoxicity curation of target assays via corresponding viability assay
        if aeid in aeids_target_assays['aeid'].values and config['enable_cytotox_filtering_by_viability_assays']:
            df = handle_cytotoxicity_based_on_viability_assays(config, str(aeid), df, viability_counterparts_aeid)

        return aeid, df

    dfs = {}
    if config['n_jobs'] != 1:
        results = Parallel(n_jobs=config['n_jobs'])(
            delayed(post_process)(aeid, aeid_path, cutoff_path, assay_info_df, aeids_target_assays, viability_counterparts_aeid) for
            aeid, aeid_path, cutoff_path in output_paths)
    else:
        results = []
        for aeid, aeid_path, cutoff_path in output_paths:
            result = post_process(aeid, aeid_path, cutoff_path, assay_info_df, aeids_target_assays,
                                  viability_counterparts_aeid)
            results.append(result)

    for aeid, df in results:
        dfs[aeid] = df

    df_all = pd.concat(dfs.values(), axis=0)
    cutoff_all = pd.concat([pd.read_parquet(file) for file in cutoff_paths])

    # Cytotoxicity curation of target assays via burst assays
    if config['enable_cytotox_filtering_by_burst_assays']:
        df_all = cytotoxicity_curation_with_burst_assays(config, df_all, aeids_target_assays)

    plot_omit_flags(df_all, aeids_sorted)

    return df_all, cutoff_all


def plot_omit_flags(df_all, aeids_sorted):
    """
    Plot overview of omit flags.
    """
    omit_flags = df_all['omit_flag'].value_counts(dropna=False).reset_index()
    omit_flags.columns = ["omit_flag", "count"]

    labels = omit_flags["omit_flag"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))

    colors = ['#78AB46', '#FFCC66', '#FF0000']

    ax.pie(omit_flags["count"], labels=labels, colors=colors, autopct='%1.1f%%', wedgeprops=dict(width=0.5, edgecolor='w'), startangle=-40)

    # Inner circle
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    fig.legend(labels, loc="center", fontsize=12)

    ax.set_title("Data curation")

    plt.savefig(os.path.join(METADATA_SUBSET_DIR_PATH, "omit_flags.png"), dpi=1200)
    plt.close()


def ice_curation_adding_omit_flags(aeid, assay_info_df, df):
    """
    Add omit flags based on ICE curation.
    """
    # Curve/assay-based curation:
    # For NovaScreen (NVS) cell-free biochemical assays, any active calls with less than 50% efficacy.
    condition_assay = assay_info_df['assay_component_endpoint_name'].str.startswith('NVS_') & (
                assay_info_df['cell_format'] == 'cell-free')
    if aeid in assay_info_df[condition_assay]['aeid'].values:
        condition_omit_flag = df['top'] < 0.5  # NovaScreen (NVS) cell-free biochemical assays normalized as percent_activity
        df.loc[condition_omit_flag, 'omit_flag'] = "OMIT (assay-based)"

    # For NCATS Tox21 assays, any active calls where only the lowest concentration tested exceeded the assay activity cutoff threshold and the best-fit curve was a gain-loss model.
    # Note: NCATS Tox21 assays not found

    # Any down-direction (i.e., inhibition, antagonism, loss-of-signal, etc.) active call where the best-fit curve was a gain-loss model.
    if aeid in assay_info_df[assay_info_df['signal_direction'] == 'loss']['aeid'].values:
        condition_omit_flag = df['best_aic_model'].isin(
            ['gnls', 'gnls2'])  # gnls and gnls2 both model loss of activity for increasing concentrations
        df.loc[condition_omit_flag, 'omit_flag'] = "OMIT (assay-based)"
    concs = df['conc'].apply(json.loads)

    # Any active call where the best-fit curve was a gain-loss model and AC50 that was extrapolated below the testing concentration range.
    min_concs = list(map(min, concs))
    condition_omit_flag = df['best_aic_model'].isin(['gnls', 'gnls2']) & (df['ac50'] < min_concs)
    df.loc[condition_omit_flag, 'omit_flag'] = "OMIT (assay-based)"

    # Any active concentration-response curve where the AC50 was extrapolated to a concentration above the tested concentration range.
    max_concs = list(map(max, concs))
    condition_omit_flag = df['ac50'] > max_concs
    df.loc[condition_omit_flag, 'omit_flag'] = "OMIT (assay-based)"

    # Any active call with flags 11 and 16 from the tcpl pipeline, which suggest marginal efficacy and likely overfitting.
    # Note: fitc flags 11 and 16 not availbale in the new (py)tcpl v3.0. Map new quality warning flags

    # ICE omits for the cHTS data set entire assay endpoints (already done in pipeline setup by subset aeids):

    # Chemical QC-based curation:
    destination_path = os.path.join(METADATA_DIR_PATH, f"compounds_qc_omit.parquet.gzip")
    compounds_qc_omit_df = pd.read_parquet(destination_path)
    condition_omit_flag = df['dsstox_substance_id'].isin(compounds_qc_omit_df['dsstox_substance_id'])
    df.loc[condition_omit_flag, 'omit_flag'] = "OMIT (chemical-based)"


def handle_cytotoxicity_based_on_viability_assays(config, aeid, df, viability_counterparts_aeid):
    """"
    Handle cytotoxicity based on viability assays.
    """
    # For cell based assays, try finding matching viability/burst/background assays and correct hitc where
    # AC50 in the former > than AC50 in the latter (we can adjust boundaries around that).
    if aeid in viability_counterparts_aeid:
        viability_assay_endpoint_aeid = viability_counterparts_aeid[aeid]
        viability_assay_endpoint_path = os.path.join(OUTPUT_DIR_PATH, f"{viability_assay_endpoint_aeid}{FILE_FORMAT}")
        viability_assay_endpoint = pd.read_parquet(viability_assay_endpoint_path)
        df.loc[:, 'viability_aeid'] = viability_assay_endpoint_aeid
        df['cytotox_ref'] = "viability"
        for index, row in df.iterrows():
            compound = row['dsstox_substance_id']
            acc_target = row['acc']
            hitcall_target = row['hitcall']
            # Find the corresponding compounds in viability_assay_endpoint
            matching_row = viability_assay_endpoint[viability_assay_endpoint['dsstox_substance_id'] == compound]

            if not matching_row.empty and hitcall_target != 0:
                acc_viability = matching_row['acc'].iloc[0]
                hitcall_viability = matching_row['hitcall'].iloc[0]

                # Estimate probability as the probability that acc_viability < acc_target, where 0 = not cytotox and 1 = cytotoxic
                diff = acc_viability - acc_target
                # Check if not NaN (if acc not defined -> no cytotoxicity interference reevaluation required)
                if diff == diff:
                    # Variance in the difference is variance of target acc + variance of cytotox acc
                    # We don't know variance of acc so use 0.3 as a generous estimate based on Watt and Judson 2018
                    var = (10 ** 0.3) ** 2 + (10 ** 0.3) ** 2

                    # This gives the probability that diff > 0 based on the cumulative probability distribution with mean = 0 and var = var
                    cytotoxicity_confounding_prob = norm.cdf(diff, loc=0, scale=var ** 0.5)

                    is_monotonic = matching_row['best_aic_model'].iloc[0] not in ['gnls', 'gnls2']
                    cytotoxicity_confounding_prob *= hitcall_viability * is_monotonic
                    prob_cytotoxicity_corrective = (1 - cytotoxicity_confounding_prob)

                    df.at[index, 'hitcall_c'] = hitcall_target * prob_cytotoxicity_corrective
                    df.at[index, 'cytotox_prob'] = cytotoxicity_confounding_prob
                    df.at[index, 'cytotox_acc'] = acc_viability

    return df


def compute_cytotoxicity_from_burst_assays(config, burst_assays):
    """
    Compute cytotoxicity from burst assays.
    """
    threshold_cytotox_min_tested = config['threshold_cytotox_min_test']
    # min_test = int(0.8 * len(burst_assays)) if threshold_cytotox_min_tested <= 1 else threshold_cytotox_min_tested
    burst_assays = burst_assays[~burst_assays['best_aic_model'].isin(['gnls', 'gnls2'])]
    hitc_num = config['threshold_cytotox_hitc_num']

    def compute_metrics(x):
        med = np.nanmedian(np.log10(x.loc[x['hitcall'] >= hitc_num, 'acc']))
        mad = get_mad(np.log10(x.loc[x['hitcall'] >= hitc_num, 'acc']))
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
    # burst_assays = burst_assays.drop(columns=['burstpct'])
    path = os.path.join(METADATA_DIR_PATH, f"cytotox_{FILE_FORMAT}")
    burst_assays.to_parquet(path, compression='gzip')
    path = os.path.join(METADATA_DIR_PATH, f"cytotox_.csv")
    burst_assays.to_csv(path)

    return burst_assays


# def group_by_aeid(aeid, df_all):
#     aeid_df = df_all[df_all['aeid'] == aeid]
#     output_file = os.path.join(OUTPUT_DIR_PATH, f'{aeid}{FILE_FORMAT}')
#     aeid_df.to_parquet(output_file, compression='gzip')
#
#     # # Replace the original file with the temporary file
#     # os.replace(output_file, os.path.join(OUTPUT_DIR_PATH, f'{aeid}{FILE_FORMAT}'))


def group_by_aeids(df_all):
    """
    Group by aeids and save to parquet files.
    """
    aeids = df_all['aeid'].dropna().unique()
    num_aeids = len(aeids)

    def process_aeid(i, aeid):
        print(f"{i + 1}/{num_aeids}: {aeid}")
        aeid_df = df_all[df_all['aeid'] == aeid]
        output_file = os.path.join(OUTPUT_DIR_PATH, f'{aeid}{FILE_FORMAT}')
        aeid_df.to_parquet(output_file, compression='gzip')

    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    max_workers = max(os.cpu_count() * 2 - 1, 1)
    print(f"max_workers: {max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, aeid in enumerate(aeids):
            executor.submit(process_aeid, i, aeid)


def cytotoxicity_curation_with_burst_assays(config, df_all, aeids_target_assays):
    """
    Cytotoxicity curation of target assays via burst assays.
    """
    path = os.path.join(METADATA_DIR_PATH, f"assay_component_endpoint{FILE_FORMAT}")
    df = pd.read_parquet(path)
    burst_assays_aeids = df[df['burst_assay'] == 1]['aeid']

    target_assays_df = df_all[df_all['aeid'].isin(aeids_target_assays['aeid'])].reset_index(drop=True)
    non_target_assays_df = df_all[~df_all['aeid'].isin(aeids_target_assays['aeid'])].reset_index(drop=True)
    burst_assays = df_all[df_all['aeid'].isin(burst_assays_aeids.values)].reset_index(drop=True)

    if not burst_assays.empty:
        # Some names have changed in the new tcpl version 3.0, e.g. cytotox_median_um = cyto_pt_um
        cytotox_df = compute_cytotoxicity_from_burst_assays(config, burst_assays)

        # Plot if cytotox_df data is complete
        check_cytotoxicity_completeness(cytotox_df)

        # Plot overview of cytotoxicity data
        plot_overview_of_cytotoxicity_data(cytotox_df)

        # Split active and inactive cases in target assays
        is_active = target_assays_df["hitcall"] > 0
        active_cases = target_assays_df[is_active].reset_index(drop=True)
        inactive_cases = target_assays_df[~is_active].reset_index(drop=True)

        print(f"Active cases in target assays: {len(active_cases)}")
        print(f"Inactive cases in target assays: {len(inactive_cases)}")

        # Merge active_cases with cytotox info
        active_cases = active_cases.merge(cytotox_df, on="dsstox_substance_id", how="left")

        # Flag different cases, active_cases["cytotox_ref"] != 'burst', -> already handled by matching viability assays
        rest_to_handle = active_cases["cytotox_ref"] == 'burst'
        active_cases.loc[rest_to_handle] = cytotox_adjustment_for_rest_of_cases_based_on_burst_assays(active_cases.loc[rest_to_handle])

        # Print summary of flags
        print_counts(active_cases)

        # Count
        count_hitc = pd.DataFrame({'hitcall': ["inactive", "active"], 'count': [(target_assays_df["hitcall"] == 0).sum(), (target_assays_df["hitcall"] > 0).sum()]})
        count_cytotox_ref = active_cases["cytotox_ref"].value_counts().reset_index()
        count_cytotox_ref.columns = ["cytotox_ref", "count"]

        # Plot an overview of the flagging and hit calls
        # plot_overview_cytotoxicity_flagging_and_hitcalls(count_cytotox_ref, count_hitc)

        # Combine active and inactive data frames and vaibility/burst assays
        df_all_c = merge_all_data(active_cases, inactive_cases, non_target_assays_df)
        return df_all_c
    else:
        return df_all


def print_counts(active_cases):
    flag_counts = active_cases["cytotox_ref"].value_counts(dropna=False)
    print(flag_counts)


def check_cytotoxicity_completeness(cytotox):
    missing_vals = cytotox["cyto_pt_um"].isna()
    if missing_vals.sum() == 0:
        print("No incomplete cytotoxicity data")
    else:
        cytotox_incomplete = cytotox[missing_vals]
        print(f"Incomplete cytotoxicity data; nrows={cytotox_incomplete.shape[0]}")
        cytotox.loc[missing_vals, "cyto_pt_um"] = cytotox["cyto_pt_um"].median()


def merge_all_data(active_cases, inactive_cases, non_target_assays_df):
    # Get common columns between df and df_all_inact
    common_cols = list(set(active_cases.columns) & set(inactive_cases.columns))
    df_common = active_cases[common_cols]
    inactive_cases = inactive_cases[common_cols]
    non_target_assays_df = non_target_assays_df[common_cols]
    # Concatenate the two data frames with only the common columns and save
    df_all_c = pd.concat([df_common, inactive_cases, non_target_assays_df], ignore_index=True)
    df_all_c['hitcall_c'] = df_all_c['hitcall_c'].astype(float)
    # folder_path = os.path.join(DATA_DIR_PATH, "merged", "output_cytotox_ref")
    # os.makedirs(folder_path, exist_ok=True)
    # file_path = os.path.join(folder_path, f"{0}{FILE_FORMAT}")
    # df_all_c.to_parquet(file_path, compression='gzip')
    return df_all_c


def plot_overview_cytotoxicity_flagging_and_hitcalls(count_cytotox_ref, count_hitacc, count_hitc):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    axes = axes.flatten()
    titles = ["Hitcalls in target assay endpoints", "Cytotoxicity flags for active hitcalls in target assay endpoints", "Hitcalls in target assay endpoints after cytotoxicity filtering"]
    x_labels = ["Hitcalls", "Cytotoxicity flag", "Hitcalls"]
    y_labels = ["Count"] * 3

    flag_lables = ["cytotoxic" if flag.startswith("cytotoxic") else
                   "non_cytotoxic" if flag.startswith("non_cytotoxic") else
                   "inconclusive"
                   for flag in count_cytotox_ref['cytotox_ref']]

    colors_list = [["red", "blue"], None, ["red", "blue"]]
    categories_list = [count_hitc["hitcall"].unique(), count_cytotox_ref["cytotox_ref"].unique(), count_hitc["hitcall"].unique()]
    categories_list = [[str(value) for value in inner_list] for inner_list in categories_list]
    labels_list = [None, flag_lables, None]
    values_list = [count_hitc["count"], count_cytotox_ref["count"],
                   [  # inactive with cytotox filtering
                       count_hitc[(count_hitc['hitcall'] == "inactive")]['count'].iloc[0] +
                       count_hitacc[(count_hitacc['ctx_acc'] == "cytotoxic")]['count'].iloc[0],
                       # active with cytotox filtering
                       count_hitacc[(count_hitacc['ctx_acc'] == "non_cytotoxic")]['count'].iloc[0]]]

    print(values_list)
    for i, (ax, values, categories, colors, title, x_label, y_label, labels) in enumerate(
            zip(axes, values_list, categories_list, colors_list, titles, x_labels, y_labels, labels_list)):

        sorted_values, sorted_categories = zip(*sorted(zip(values, categories), key=lambda x: x[1]))

        # ax.set_yticks([])
        # Create a stacked bar chart for the second subplot
        if i == 1:
            groups = ["cytotoxic", "non_cytotoxic", "inconclusive"]

            cytotoxic = []
            non_cytotoxic = []
            inconclusive = []

            for category, value in zip(sorted_categories, sorted_values):
                if category.startswith(groups[0]):
                    cytotoxic.append(value)
                elif category.startswith(groups[1]):
                    non_cytotoxic.append(value)
                else:
                    inconclusive.append(value)

            # Calculate cumulative sums for each group
            cumulative_cytotoxic = [0] + [sum(cytotoxic[:i+1]) for i in range(len(cytotoxic))]
            cumulative_non_cytotoxic = [0] + [sum(non_cytotoxic[:i+1]) for i in range(len(non_cytotoxic))]
            cumulative_inconclusive = [0] + [sum(inconclusive[:i+1]) for i in range(len(inconclusive))]

            colors = sns.color_palette("hls", len(categories))

            # Create the stacked horizontal bar plot
            ax.barh(groups, [sum(cytotoxic), sum(non_cytotoxic), sum(inconclusive)], color=colors)
            ax.barh(groups[0], cytotoxic, left=cumulative_cytotoxic[:-1], label='cytotoxic',
                    color=colors[:len(cytotoxic)])
            ax.barh(groups[1], non_cytotoxic, left=cumulative_non_cytotoxic[:-1], label='non_cytotoxic',
                    color=colors[len(cytotoxic):len(cytotoxic) + len(non_cytotoxic)])
            ax.barh(groups[2], inconclusive, left=cumulative_inconclusive[:-1], label='inconclusive',
                    color=colors[-len(inconclusive):])

            # Add a legend for all categories
            # Create custom legend handles and labels for all categories
            legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(sorted_categories))]
            legend_labels = sorted_categories
            # Add a legend for all categories
            ax.legend(legend_handles, legend_labels, loc='center right', framealpha=0.3)

        else:
            ax.barh(sorted_categories, sorted_values, color=colors, label=labels)  # Todo: color=colors, label=labels # does not work yet
            ax.set_yticks(categories)
            ax.set_xlim(0, 1000000)  # Todo: use dynamic value
            if labels is not None:
                ax.legend(sorted_categories, fontsize='small')

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)



    plt.tight_layout()
    plt.savefig(os.path.join(METADATA_SUBSET_DIR_PATH, "counts_hitc_cytotox_ref.png"), dpi=1200)
    plt.close()


def cytotox_adjustment_for_rest_of_cases_based_on_burst_assays(df):
    """
    Cytotoxicity adjustment for rest of cases based on burst assays.
    """
    hitcall_target = df['hitcall']
    acc_target = df['acc']
    acc_cytotox = df['cyto_pt_um']
    mad_cytotox = df['mad']
    nhit_over_ntested = df['burstpct']

    # Estimate probability as the probability that acc_viability < acc_target, where 0 = not cytotox and 1 = cytotoxic
    diff = acc_cytotox - acc_target
    # Variance in the difference is variance of target acc + variance of cytotox acc
    # Convert MAD to stdev, assuming normal dist (https://blog.arkieva.com/relationship-between-mad-standard-deviation/)
    cytotox_sd = (10 ** mad_cytotox) / ((2 / np.pi) ** 0.5)
    # We don't know variance of target acc so use 0.3 as a generous estimate based on Watt and Judson 2018
    var = (10 ** 0.3) ** 2 + cytotox_sd ** 2
    # This gives the probability that diff > 0 based on the cumulative probability distribution with mean = 0 and var = var
    cytotoxicity_confounding_prob = norm.cdf(diff, loc=0, scale=var ** 0.5)
    # Multiply by nhit_over_ntested
    cytotoxicity_confounding_prob *= nhit_over_ntested
    # cytotoxicity_confounding_prob = "prob that cytotoxicity kicks in at lower concentrations than specific toxicity" *  "ratio where cytotoxicity occurs below the max tested concentration"
    # "new hitcall" = "orignial hitcall" * (1 - cytotoxicity_confounding_prob)
    prob_cytotoxicity_corrective = (1 - cytotoxicity_confounding_prob)
    mask_non_nan = ~np.isnan(prob_cytotoxicity_corrective)
    df.loc[mask_non_nan, 'hitcall_c'] = hitcall_target * prob_cytotoxicity_corrective
    df['cytotox_prob'] = cytotoxicity_confounding_prob
    df['cytotox_acc'] = acc_cytotox

    return df


def plot_overview_of_cytotoxicity_data(cytotox):
    """
    Plot overview of cytotoxicity data.
    """
    fig, axes = plt.subplots(5, 1, figsize=(10, 10))
    axes = axes.flatten()
    titles = ["Median (ac50) cytotoxicity (log10 µM) ", "MAD (ac50) cytotoxicity (log10 µM)", "Lower bound: Median - 3 * global MAD (log10 µM)", "# tested", "# active"]
    x_labels = ["cytotox_median_raw", "cytotox_mad", "cytotox_lower_bound", "ntested", "nhit"]
    y_labels = ["Density"] * 5
    x_ranges = [(0, 2), (0, 2), (0, 3), (0, 100), (0, 100)]
    y_ranges = [None, None, None, None, None]
    bins_list = [100, 100, 500, 100, 100]
    legend_notes = ["(all values shown)",
                    "(values > 1 not shown)",
                    "(values > 120 not shown)",
                    "(all values shown)",
                    "(all values shown)"]
    data = [cytotox["cyto_pt"], cytotox["mad"], np.log10(cytotox["lower_bnd_um"]), cytotox["ntst"], cytotox["nhit"]]
    for i, (ax, data, bins, title, x_label, y_label, legend_note, x_range, y_range) in enumerate(
            zip(axes, data, bins_list, titles, x_labels, y_labels, legend_notes, x_ranges, y_ranges)):
        ax.hist(data, bins=bins, color="cornflowerblue")  # label=legend_note
        ax.set_title(title)  # legend_note
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_range is not None:
            ax.set_xlim(x_range)
        # if y_range is not None:
        #     ax.set_xlim(y_range)
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(METADATA_SUBSET_DIR_PATH, "cytotox_overview_alldata.png"), dpi=1200)
    plt.close()
