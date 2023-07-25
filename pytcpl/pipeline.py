import cProfile
import os

import numpy as np
import pandas as pd
import json

from mc4_mthds import mc4_mthds
from mc5_mthds import mc5_mthds
from pipeline_helper import starting, elapsed, get_mc5_data, ensure_all_new_db_tables_exist, load_config, \
    export_data, drop_tables, ROOT_DIR, get_cutoff, track_fitted_params
from pytcpl.pipeline_helper import read_log_file
from tcpl_fit import tcpl_fit
from tcpl_hit import tcpl_hit
from tcpl_load_data import tcpl_load_data
from tcpl_mthd_load import tcpl_mthd_load
from tcpl_output import tcpl_output
from tcpl_write_data import tcpl_write_data
from tcpl_write_data import tcpl_append

config = load_config()["pytcpl"]

# warnings.filterwarnings("ignore")
# warnings.filterwarnings("error", category=RuntimeWarning)


def pipeline():
    aeid = config['aeid']
    start_time = starting(f"Starting pipeline with assay id {aeid}")
    drop_tables(config["new_table_names"]) # uncomment if you want to remove the specified pipeline tables from the db
    ensure_all_new_db_tables_exist()
    df = tcpl_load_data(lvl=3, fld='aeid', ids=aeid)
    print(f"Loaded L3 with ({df.shape[0]} rows) >> {elapsed(start_time)}")
    get_bmad = tcpl_mthd_load(lvl=4, aeid=aeid)
    for mthd in list(get_bmad['mthd'].values): #+['onesd.aeid.lowconc.twells']
        df = mc4_mthds(mthd)(df)

    df = tcpl_fit(dat=df, fit_models=config["fit_models"], fit_strategy=config["fit_strategy"],
                  bidirectional=config["bidirectional"], parallelize=config["parallelize"], n_jobs=config["n_jobs"],
                  test=config["test"])

    track_fitted_params()

    print(f"Curve-fitted {df.shape[0]} series, with {len(config['fit_models'])} fit models > {elapsed(start_time)}")

    bmad = df["bmad"].iloc[0]
    cutoff = get_cutoff(aeid=aeid, bmad=bmad)
    tcpl_append(pd.DataFrame({"aeid": [aeid], "bmad": [bmad], "cutoff": [cutoff]}), "cutoffs")


    dat = tcpl_hit(df, config["fit_strategy"], cutoff, parallelize=config["parallelize"], n_jobs=config["n_jobs"])

    dat.loc[:, 'concentration_unlogged'] = dat['concentration_unlogged'].apply(json.dumps)
    dat.loc[:, 'response'] = dat['response'].apply(json.dumps)
    dat.loc[:, 'fitparams'] = dat['fitparams'].apply(json.dumps)
    tcpl_append(dat[config["out_cols"]], "out")

    print(f"Computed L5 hitcall parameters with {dat.shape[0]} rows >> {elapsed(start_time)}")
    if config["store"]:
        export_data(dat, path=config["export_path"], folder="mc5", id=aeid)
    print(f"Stored L5 with {dat.shape[0]} rows to db >> {elapsed(start_time)}")

    export_cols = ["aeid", "spid", "fit_model", "cutoff", "hitcall", "ac50", "bmd", "top"]

    dat = dat[export_cols]
    dat = tcpl_output(dat)
    dat = dat.rename(columns={"dsstox_substance_id": "dtxsid"})
    if config["do_export"]:
        df_export = dat[['dtxsid', "chit"]]
        export_data(df_export, path=config["export_path"], folder="out", id=aeid)
        print(f"Done export >> {elapsed(start_time)}\n")
    return dat
    print(f"Pipeline done >> {elapsed(start_time)}\n")




if __name__ == '__main__':
    if config["profile"]:
        with cProfile.Profile() as pr:
            pipeline()
        pr.dump_stats(os.path.join(ROOT_DIR, f'profile/pipeline.prof') )
        print("Profiling complete")  # Use `snakeviz pytcpl/profile/pipeline.prof` to view profile in browser
    else:
        pipeline()
