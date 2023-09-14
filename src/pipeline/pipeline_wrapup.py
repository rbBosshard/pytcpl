import os
import sys
import time

import pandas as pd

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.pipeline.pipeline_constants import FILE_FORMAT, METADATA_SUBSET_DIR_PATH

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

from src.pipeline.pipeline_helper import load_config, init_config, init_aeid
from src.pipeline.pipeline_wrapup_helper import merge_all_outputs, compute_cytotoxicity_info, merge_all_outputs_and_save, \
    groupb_by_compounds


def correct_cytotoxic_hitcalls(config, df_all):
    path = os.path.join(METADATA_SUBSET_DIR_PATH, f"cytotox_{FILE_FORMAT}")
    cytotox = pd.read_parquet(path)
    missing_vals = cytotox["cytotox_median_um"].isna()
    if missing_vals.sum() == 0:
        print("No incomplete cytotoxicity data")
    else:
        cytotox_incomplete = cytotox[missing_vals]
        print(f"Incomplete cytotoxicity data; nrows={cytotox_incomplete.shape[0]}")
        print(cytotox_incomplete.head())
        median_value = cytotox["cytotox_median_um"].median()
        cytotox.loc[missing_vals, "cytotox_median_um"] = median_value

    # Plot overview of cytotoxicity data
    plt.figure(figsize=(10, 10))
    plt.subplot(3, 2, 1)
    plt.hist(cytotox["cytotox_median_raw"], color="blue", bins=500, alpha=0.7)
    plt.title("cytotox_median_raw")

    plt.subplot(3, 2, 2)
    plt.hist(cytotox["cytotox_mad"], color="blue", bins=100, alpha=0.7)
    plt.title("cytotox_mad")

    plt.subplot(3, 2, 3)
    plt.hist(cytotox["cytotox_lower_bound_um"], color="blue", bins=500, alpha=0.7)
    plt.title("cytotox_lower_bound_um")

    plt.subplot(3, 2, 4)
    plt.hist(cytotox["ntested"], color="blue", bins=50, alpha=0.7)
    plt.title("ntested")

    plt.subplot(3, 2, 5)
    plt.hist(cytotox["cytotox_median_um"], color="blue", bins=100, alpha=0.7)
    plt.title("cytotox_median_um")

    plt.subplot(3, 2, 6)
    plt.hist(cytotox["nhit"], color="blue", bins=50, alpha=0.7)
    plt.title("nhit")

    plt.tight_layout()
    plt.savefig("data/plots/cytotox_overview_alldata.pdf")
    plt.close()





def main():
    print("Started")
    start_time = time.time()

    config, _ = load_config()
    init_config(config)
    init_aeid(0)

    # remove_files_not_matching_to_aeid_list()
    df_all, cutoff_all = merge_all_outputs()
    # Potential hitcall correction based on cytotoxicity..
    df_all = compute_cytotoxicity_info(config, df_all)
    df_all = correct_cytotoxic_hitcalls(config, df_all)
    merge_all_outputs_and_save(df_all, cutoff_all)
    groupb_by_compounds(config, df_all)

    print("Finished")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
