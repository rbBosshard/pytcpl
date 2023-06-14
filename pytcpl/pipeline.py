import cProfile

import numpy as np
import pandas as pd

from mc4_mthds import mc4_mthds
from mc5_mthds import mc5_mthds
from pytcpl.pipeline_helper import starting, elapsed, get_mc5_data, ensure_all_new_db_tables_exist, load_config, \
    export_data
from tcpl_fit import tcpl_fit
from tcpl_hit import tcpl_hit
from tcpl_load_data import tcpl_load_data
from tcpl_mthd_load import tcpl_mthd_load
from tcpl_prep_otpt import tcpl_prep_otpt
from tcpl_subset_chid import tcpl_subset_chid
from tcpl_write_data import tcpl_write_data

# import warnings
# warnings.filterwarnings("ignore")
# warnings.filterwarnings("error", category=RuntimeWarning)

config = load_config()["pytcpl"]

def mc4():
    aeid = config['aeid']
    start_time = starting(f"mc4 with id {aeid}")
    df = tcpl_load_data(lvl=3, fld='aeid', ids=aeid)
    df = df.head(config["head"]) if config["test"] else df
    print(f"Loaded L3 with ({df.shape[0]} rows) >> {elapsed(start_time)}")

    get_bmad = tcpl_mthd_load(lvl=4, aeid=aeid)
    for mthd in get_bmad['mthd']:
        df = mc4_mthds(mthd)(df)

    fit_models = config["fit_models"]
    df = tcpl_fit(df, fit_models, bidirectional=["bidirectional"], force_fit=False, parallelize=config["parallelize"],
                  n_jobs=config["n_jobs"], verbose=config["verbose"])
    if config["store"]:
        export_data(df, path=config["export_path"], folder="mc4", id=aeid)
    print(f"Curve-fitted {df.shape[0]} series, with {len(fit_models)} fit models > {elapsed(start_time)}")

    tcpl_write_data(id=aeid, dat=df, lvl=4, verbose=config["verbose"])
    print(f"Stored L4 with {df.shape[0]} rows to db >> {elapsed(start_time)}")
    print("Done mc4")
    return df


def mc5(df):
    aeid = config['aeid']
    start_time = starting(f"mc5 with id {aeid}")
    if df is None:
        df = tcpl_load_data(lvl=4, fld='aeid', ids=aeid, verbose=config["verbose"])
    print(f"Loaded L4 AEID {aeid} with ({df.shape[0]} rows) >> {elapsed(start_time)}")

    assay_cutoff_methods = tcpl_mthd_load(lvl=5, aeid=aeid)["mthd"]
    bmad = df["bmad"].iloc[0]
    cutoffs = [mc5_mthds(mthd, bmad) for mthd in assay_cutoff_methods]
    cutoff = np.max(cutoffs) if len(cutoffs) > 0 else 0

    dat = get_mc5_data(aeid)
    dat = tcpl_hit(dat, cutoff, parallelize=config["parallelize"], n_jobs=config["n_jobs"], verbose=config["verbose"])

    print(f"Computed L5 hitcall parameters with {dat.shape[0]} rows >> {elapsed(start_time)}")
    if config["store"]:
        export_data(dat, path=config["export_path"], folder="mc5", id=aeid)
    tcpl_write_data(id=aeid, dat=dat, lvl=5, verbose=config["verbose"])
    print(f"Stored L5 with {df.shape[0]} rows to db >> {elapsed(start_time)}")
    print("Done mc5")


def pipeline():
    aeid = config['aeid']
    start_time = starting(f"pipeline with assay id {aeid}")
    # drop_tables(config["new_table_names"]) # uncomment if you want to remove the specified pipeline tables from the db
    ensure_all_new_db_tables_exist()
    df = mc4() if config["do_fit"] else pd.read_csv(config["export_path"] + f"mc4/{aeid}.csv")
    mc5(df)
    export()
    print(f"Pipeline done >> {elapsed(start_time)}")


def export():
    aeid = config['aeid']
    start_time = starting(f"Export {aeid}")
    df = tcpl_load_data(lvl=5, fld="aeid", ids=aeid, verbose=config["verbose"])
    df = tcpl_prep_otpt(df)
    df = tcpl_subset_chid(dat=df, flag=False)
    export_data(df, path=config["export_path"], folder="out", id=aeid)
    print(f"Done export >> {elapsed(start_time)}")


if __name__ == '__main__':
    if config["profile"]:
        with cProfile.Profile() as pr:
            pipeline()
        pr.dump_stats(f'profile/pipeline.prof')
        print("Profiling complete")  # Use `snakeviz pytcpl/profile/pipeline.prof` to view profile in browser
    else:
        pipeline()
