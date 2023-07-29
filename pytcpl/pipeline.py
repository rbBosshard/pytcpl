import cProfile
import json
import os

from pipeline_helper import load_config, \
    ROOT_DIR, print_, prolog, get_efficacy_cutoff_and_append, tcpl_delete, launch
from pipeline_helper import load_raw_data_from_db, tcpl_append, export_as_csv, status
from tcpl_fit import tcpl_fit
from tcpl_hit import tcpl_hit


def pipeline(config, confg_path):
    
    aeid_list = launch(config, confg_path)

    for aeid in aeid_list:
        prolog(aeid, config)
        df = load_raw_data_from_db(aeid=config['aeid'])
        cutoff, df = get_efficacy_cutoff_and_append(config['aeid'], df)

        df = tcpl_fit(dat=df, cutoff=cutoff, config=config)
        dat = tcpl_hit(df=df, cutoff=cutoff, config=config)

        mb_value = dat.memory_usage(deep=True).sum() / (1024 * 1024)
        mb_value = f"{mb_value:.2f} MB"
        print_(f"{status('computer_disk')} Writing output data to DB (~{mb_value})..")
        for col in ['concentration_unlogged', 'response', 'fitparams']:
            dat.loc[:, col] = dat[col].apply(json.dumps)

        tcpl_delete(config['aeid'], "output")
        tcpl_append(dat[config['output_cols']], "output")
        export_as_csv(config, dat)
        print_(f"{status('carrot')} Assay endpoint processing completed")

    print(f"\n{status('clinking_beer_mugs')} Pipeline completed\n")
    print(f"{status('waving_hand')} Bye!")


if __name__ == '__main__':
    cnfg, cnfg_path = load_config()
    if cnfg['profile']:
        with cProfile.Profile() as pr:
            pipeline(cnfg, cnfg_path)
        pr.dump_stats(os.path.join(ROOT_DIR, "profile/pipeline.prof"))
    else:
        pipeline(cnfg, cnfg_path)
