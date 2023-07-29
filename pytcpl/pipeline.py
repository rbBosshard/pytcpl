import cProfile
import json
import os

from pipeline_helper import load_config, \
    ROOT_DIR, print_, prolog, get_efficacy_cutoff_and_append, read_aeids, tcpl_delete, check_db, launch
from pipeline_helper import load_raw_data_from_db, tcpl_append, export_as_csv
from tcpl_fit import tcpl_fit
from tcpl_hit import tcpl_hit


def pipeline(config, confg_path):
    
    aeid_list = launch(config, confg_path)

    for aeid in aeid_list:
        prolog(aeid, config)
        df = load_raw_data_from_db(aeid=config['aeid'])
        cutoff, df = get_efficacy_cutoff_and_append(config['aeid'], df)

        print_(f"Fitting curves: {df.shape[0]} concentration-response series "
               f"using {len(config['fit_models'])} different models...")
        df = tcpl_fit(dat=df, cutoff=cutoff, config=config)
        print_("Computing biological activity labels...")
        dat = tcpl_hit(df=df, cutoff=cutoff, config=config)

        mb_value = dat.memory_usage(deep=True).sum() / (1024 * 1024)
        mb_value = f"{mb_value:.2f} MB"
        print_(f"Writing output data to DB with {dat.shape[0]} rows (~{mb_value})...")
        for col in ['concentration_unlogged', 'response', 'fitparams']:
            dat.loc[:, col] = dat[col].apply(json.dumps)

        tcpl_delete(config['aeid'], "output")
        tcpl_append(dat[config['output_cols']], "output")
        export_as_csv(config, dat)
        print_("Completed processing assay endpoint.")

    print("\nPipeline completed!")


if __name__ == '__main__':
    cnfg, cnfg_path = load_config()
    if cnfg['profile']:
        with cProfile.Profile() as pr:
            pipeline(cnfg, cnfg_path)
        pr.dump_stats(os.path.join(ROOT_DIR, "profile/pipeline.prof"))
    else:
        pipeline(cnfg, cnfg_path)
