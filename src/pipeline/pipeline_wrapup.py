import os
import sys
import time

import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use('seaborn')
plt.style.use('ggplot')
# plt.style.use('classic')  # cool
# plt.style.use('bmh')
# sns.set_style('ticks')

from src.pipeline.pipeline_constants import FILE_FORMAT, METADATA_SUBSET_DIR_PATH, DATA_DIR_PATH

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

from src.pipeline.pipeline_helper import load_config, init_config, init_aeid
from src.pipeline.pipeline_wrapup_helper import merge_all_outputs, compute_cytotoxicity_info, \
    merge_all_outputs_and_save, \
    groupb_by_compounds, remove_files_not_matching_to_aeid_list


def correct_cytotoxic_hitcalls(config, df_all):
    path = os.path.join(METADATA_SUBSET_DIR_PATH, f"cytotox_{FILE_FORMAT}")
    cytotox = pd.read_parquet(path)
    missing_vals = cytotox["cyto_pt_um"].isna()
    if missing_vals.sum() == 0:
        print("No incomplete cytotoxicity data")
    else:
        cytotox_incomplete = cytotox[missing_vals]
        print(f"Incomplete cytotoxicity data; nrows={cytotox_incomplete.shape[0]}")
        print(cytotox_incomplete.head())
        median_value = cytotox["cytotox_median_um"].median()
        cytotox.loc[missing_vals, "cytotox_median_um"] = median_value

    # Plot overview of cytotoxicity data
    # Create subplots

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

    data = [cytotox["cyto_pt"],
            cytotox["mad"],
            cytotox["lower_bnd_um"],
            cytotox["ntst"],
            cytotox["cyto_pt_um"],
            cytotox["nhit"]]

    for i, (ax, data, bins, title, x_label, y_label, legend_note, x_range, y_range) in enumerate(
            zip(axes, data, bins_list, titles, x_labels, y_labels, legend_notes, x_ranges, y_ranges)):
        ax.hist(data, bins=bins)  # label=legend_note
        ax.set_title(legend_note)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        # ax.legend([legend_note])
        if x_range is not None:
            ax.set_xlim(x_range)
        if y_range is not None:
            ax.set_xlim(y_range)
        # Hide y-axis ticks
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(METADATA_SUBSET_DIR_PATH, "cytotox_overview_alldata.pdf"))
    plt.close()

    # Some names have changed in the new version but is equivalent, e.g. cytotox_median_um = cyto_pt_um
    # Active and inactive cases
    df_all_act = df_all[df_all["hitcall"] > 0]
    print(df_all_act.shape)
    df_all_inact = df_all[df_all["hitcall"] == 0]
    print(df_all_inact.shape)
    df_all_inact.loc[:, "hitc_acc"] = 0

    # Calculate acc_um and join cytotox to specific tox data from mc5
    df_all_act.loc[:, "acc_um"] = 10 ** df_all_act["acc"]
    df = df_all_act.merge(cytotox, on="dsstox_substance_id", how="left")

    # Flag different cases
    df["cytotox_flag"] = None

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

    # Flag = 50: all remaining cases. Todo: Assert if this is desired
    df.loc[df["cytotox_flag"].isna(), "cytotox_flag"] = 50

    # Print summary of flags
    flag_counts = df["cytotox_flag"].value_counts(dropna=False)
    print("active", len(df_all_act))
    print("High conf. of cytotoxicity:", flag_counts.get(10, 0))
    print("Some conf. of cytotoxicity:", flag_counts.get(11, 0))
    print("Possible cytotoxicity:", flag_counts.get(12, 0))
    print("High conf. of no cytotoxicity:", flag_counts.get(0, 0))
    print("Some conf. of no cytotoxicity:", flag_counts.get(1, 0))
    print("Unlikely cytotoxicity:", flag_counts.get(2, 0))
    print("No data:", flag_counts.get(50, 0))

    # Binary determination of cytotoxicity
    df["ctx_acc"] = "inconclusive"
    df.loc[(df["cytotox_flag"] >= 10) & (df["cytotox_flag"] != 50), "ctx_acc"] = "cytotoxic"
    df.loc[df["cytotox_flag"] <= 2, "ctx_acc"] = "not cytotoxic"

    # Calculate hitc_acc
    df["hitc_acc"] = np.where(df["ctx_acc"] == "cytotoxic", 0, df["hitcall"])

    count_hitc = pd.DataFrame({'hitcall': ["inactive", "active"],
                               'count': [(df_all["hitcall"] == 0).sum(), (df_all["hitcall"] > 0).sum()]})
    count_hitacc = df["ctx_acc"].value_counts(dropna=False).reset_index()
    count_hitacc.columns = ["ctx_acc", "count"]
    count_cytotox_flag = df["cytotox_flag"].value_counts().reset_index()
    count_cytotox_flag.columns = ["cytotox_flag", "count"]
    count_cytotox_flag["col"] = "cyan"
    count_cytotox_flag.loc[count_cytotox_flag["cytotox_flag"] >= 10, "col"] = "magenta"
    count_cytotox_flag.loc[count_cytotox_flag["cytotox_flag"] == 50, "col"] = "grey"

    # Plot an overview of the flagging and hit calls
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    axes = axes.flatten()

    titles = ["Hit calls in relevant target subset of assay endpoints", "Cytotox. flags for positive hit calls",
              "Positive hit calls, filtered for cytotox.", "Hit calls after cytotox filtering"]
    x_labels = ["Hitcall", "Cytotox. flag", "Hitcall, ct filtered", "Hitcall"]
    y_labels = ["Count"] * 4

    # Define custom qualitative colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
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

    values_list = [count_hitc["count"],
                   count_cytotox_flag["count"],
                   count_hitacc["count"],
                   [  # inactive with cytotox filtering
                       count_hitc[(count_hitc['hitcall'] == "inactive")]['count'].iloc[0] -
                       count_hitacc[(count_hitacc['ctx_acc'] == "cytotoxic")]['count'].iloc[0],
                       # active with cytotox filtering
                       count_hitacc[(count_hitacc['ctx_acc'] == "not cytotoxic")]['count'].iloc[0]]]
    labels_list = [None, ["inconclusive", "cytotoxic", "not cytotoxic"], None, None]

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

    # # Combine positive and negative hit call data frames
    df_all_inact.loc[:, "cytotox_flag"] = None
    df_all_inact.loc[:, "hitc_acc"] = 0

    # Get common columns between df and df_all_inact
    common_cols = list(set(df.columns) & set(df_all_inact.columns))
    df_common = df[common_cols]
    df_all_inact_common = df_all_inact[common_cols]

    # Concatenate the two data frames with only the common columns and save
    df_all_c = pd.concat([df_common, df_all_inact_common], ignore_index=True)
    df_all_c['hitc_acc'] = df_all_c['hitc_acc'].astype(float)

    folder_path = os.path.join(DATA_DIR_PATH, "merged", "output_cytotox_flagged")
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{0}{FILE_FORMAT}")
    df_all_c.to_parquet(file_path, compression='gzip')


def main():
    print("Started")
    start_time = time.time()

    config, _ = load_config()
    init_config(config)
    init_aeid(0)

    # remove_files_not_matching_to_aeid_list(delete=False)
    df_all, cutoff_all = merge_all_outputs()
    # Potential hitcall correction based on cytotoxicity..
    burst_assays = compute_cytotoxicity_info(config, df_all)
    df_all = correct_cytotoxic_hitcalls(config, df_all)
    # merge_all_outputs_and_save(df_all, cutoff_all)
    # groupb_by_compounds(config, df_all)

    print("Finished")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
