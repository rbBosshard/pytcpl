import os
import sys
import time

import pandas as pd
pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt

plt.style.use('ggplot')

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

from src.pipeline.pipeline_helper import load_config, init_config, init_aeid
from src.pipeline.pipeline_wrapup_helper import ice_curation_and_cytotoxicity_handling, \
    save_merged, groupb_by_compounds, group_by_aeids


def main():
    """
    Main function for the pipeline wrapup.
    """
    print("Started")
    start_time = time.time()

    config, _ = load_config()
    init_config(config)
    init_aeid(0)

    # remove_files_not_matching_to_aeid_list(delete=False)
    df_all, cutoff_all = ice_curation_and_cytotoxicity_handling(config)
    group_by_aeids(df_all)
    groupb_by_compounds(config, df_all)
    save_merged(df_all, cutoff_all)

    print("Finished")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
