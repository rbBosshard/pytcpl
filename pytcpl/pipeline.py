import cProfile
import os

from pipeline_helper import load_config, \
    ROOT_DIR, prolog, epilog, get_efficacy_cutoff_and_append, launch, load_raw_data_from_db, export_as_csv, goodbye, \
    write_output_data_to_db
from processing import processing


def pipeline(config, confg_path):
    aeid_list = launch(config, confg_path)
    for aeid in aeid_list:
        prolog(aeid, config)
        df = load_raw_data_from_db(config['aeid'])
        cutoff, df = get_efficacy_cutoff_and_append(config['aeid'], df)
        df = processing(df, cutoff, config)
        write_output_data_to_db(config, df)
        export_as_csv(config, df)
        epilog()
    goodbye()


if __name__ == '__main__':
    cnfg, cnfg_path = load_config()
    if cnfg['profile']:
        with cProfile.Profile() as pr:
            pipeline(cnfg, cnfg_path)
        pr.dump_stats(os.path.join(ROOT_DIR, "profile/pipeline.prof"))
    else:
        pipeline(cnfg, cnfg_path)
