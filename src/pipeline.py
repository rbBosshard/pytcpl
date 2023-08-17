import cProfile
import os

from utils.constants import ROOT_DIR, PROFILER_PATH, LOG_DIR_PATH
from utils.pipeline_helper import load_config, prolog, launch, fetch_raw_data, bye, write_output, epilog
from utils.process import process


def pipeline(config, confg_path):
    instance_id, instances_total, aeid_list, logger = launch(config, confg_path)
    for aeid in aeid_list:
        try:
            prolog(aeid, instance_id)
            df = fetch_raw_data()
            df = process(df, config, logger)
            write_output(df)
            epilog()
            raise Exception("klsdf")
        except Exception as e:
            error_file_path = os.path.join(LOG_DIR_PATH, f"errors_{instance_id}.log")
            with open(error_file_path, "a") as f:
                err_msg = f"Assay endpoint with aeid={aeid} failed: {e}"
                logger.error(err_msg)
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
