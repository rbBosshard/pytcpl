import cProfile
import os

from utils.constants import ROOT_DIR, PROFILER_PATH, ERROR_PATH
from utils.pipeline_helper import load_config, prolog, launch, load_raw_data, export, bye, store_output_in_db, epilog
from utils.process import process


def pipeline(config, confg_path):
    aeid_list = launch(config, confg_path)
    for aeid in aeid_list:
        try:
            prolog(config, aeid)
            df, cutoff = load_raw_data()
            df = process(df, cutoff, config)
            store_output_in_db(df)
            export(df)
            epilog()
        except Exception as e:
            with open(ERROR_PATH, "a") as f:
                err_msg = f"Assay endpoint with aeid={aeid} failed: {e}"
                print(err_msg)
                print(err_msg, file=f)
    bye()


if __name__ == '__main__':
    cnfg, cnfg_path = load_config()
    if cnfg['enable_profiling']:
        with cProfile.Profile() as pr:
            pipeline(cnfg, cnfg_path)
        pr.dump_stats(os.path.join(ROOT_DIR, PROFILER_PATH))
    else:
        pipeline(cnfg, cnfg_path)
