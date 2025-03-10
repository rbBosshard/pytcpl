import cProfile
import traceback
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

from src.pipeline.pipeline_constants import ROOT_DIR, PROFILER_PATH, LOG_DIR_PATH
from src.pipeline.pipeline_helper import load_config, prolog, launch, fetch_raw_data, bye, write_output, epilog
from src.pipeline.pipeline_core import process


def pipeline(config, confg_path):
    """
    Execute the data processing pipeline for a list of assay endpoints.

    This function executes the complete data processing pipeline for a list of assay endpoints. The pipeline includes
    data retrieval, preprocessing, modeling, and output writing for each assay endpoint.

    Args:
        config (dict): A dictionary containing configuration parameters for the pipeline.
        confg_path (str): The path to the configuration file.

    Returns:
        None

    Note:
    This function iterates over a list of assay endpoint IDs and performs the following steps for each endpoint:
    1. Launches the processing instance and prepares the logging.
    2. Retrieves raw data for the given assay endpoint.
    3. Processes the raw data according to the provided configuration.
    4. Writes the processed output data.
    5. Finalizes the processing for the endpoint.
    If an exception occurs during processing, an error message is logged and written to an error file.

    """
    instance_id, instances_total, aeid_list, logger = launch(config, confg_path)
    for aeid in aeid_list:
        try:
            assay_endpoint_info = prolog(aeid, instance_id)
            df = fetch_raw_data()
            df = process(assay_endpoint_info, df, config, logger)
            write_output(df)
            epilog()
        except Exception as e:
            traceback_info = traceback.format_exc()
            error_file_path = os.path.join(LOG_DIR_PATH, f"errors_{instance_id}.log")
            with open(error_file_path, "a") as f:
                err_msg = f"Assay endpoint with aeid={aeid} failed: {e}"
                logger.error(err_msg)
                print(err_msg, file=f)
                print(traceback_info, file=f)
    bye()


if __name__ == '__main__':
    cnfg, cnfg_path = load_config()
    if cnfg['enable_profiling']:
        with cProfile.Profile() as pr:
            pipeline(cnfg, cnfg_path)
        pr.dump_stats(os.path.join(ROOT_DIR, PROFILER_PATH))
    else:
        pipeline(cnfg, cnfg_path)
