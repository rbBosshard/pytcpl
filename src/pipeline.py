import cProfile
import os

from constants import ROOT_DIR, PROFILER_PATH
from pipeline_helper import load_config, prolog, epilog, launch, load_raw_data, export, bye, store_output_in_db
from processing import processing


def pipeline(config, confg_path):
    aeid_list = launch(config, confg_path)
    for aeid in aeid_list:
        prolog(config, aeid)
        df, cutoff = load_raw_data()
        df = processing(df, cutoff, config)
        store_output_in_db(df)
        export(df)
        epilog()
    bye()


if __name__ == '__main__':
    cnfg, cnfg_path = load_config()
    if cnfg['enable_profiling']:
        with cProfile.Profile() as pr:
            pipeline(cnfg, cnfg_path)
        pr.dump_stats(os.path.join(ROOT_DIR, PROFILER_PATH))
    else:
        pipeline(cnfg, cnfg_path)
