import cProfile
import warnings

import numpy as np
import pandas as pd

from mc4_mthds import mc4_mthds
from mc5_mthds import mc5_mthds
from pytcpl.pipeline_helper import starting, elapsed, get_mc5_data, ensure_all_new_db_tables_exist
from tcpl_fit import tcpl_fit
from tcpl_hit import tcpl_hit
from tcpl_load_data import tcpl_load_data
from tcpl_mthd_load import tcpl_mthd_load
from tcpl_prep_otpt import tcpl_prep_otpt
from tcpl_subset_chid import tcpl_subset_chid
from tcpl_write_data import tcpl_write_data

# warnings.filterwarnings("ignore")
warnings.filterwarnings("error", category=RuntimeWarning)

aeid = 80
head = 20
test = 0
parallelize = 1
verbose = 0
profile = 1
do_fit = 1
bidirectional = True
ml = 1
plot = 1
export_path = "export/"
new_table_names = ["mc4_", "mc4_agg_", "mc4_param_", "mc5_", "mc5_param_"]
fit_models = ["cnst", "poly1", "poly2", "pow", "exp2", "exp3", "exp4", "exp5", "hill", "gnls"]
# fit_models = ["cnst", "poly1"]
# fit_models = ["cnst", "poly1", "poly2", "exp2"]


def mc4():
    start_time = starting(f"mc4 with id {aeid}")
    df = tcpl_load_data(lvl=3, fld='aeid', val=aeid)
    df = df.head(head) if test else df
    print(f"Loaded L3 AEID {aeid} with ({df.shape[0]} rows) >> {elapsed(start_time)}")

    get_bmad = tcpl_mthd_load(lvl=4, aeid=aeid)
    for mthd in get_bmad['mthd']:
        df = mc4_mthds(mthd)(df)

    df = tcpl_fit(df, fit_models, bidirectional, force_fit=False, parallelize=parallelize, verbose=verbose)
    df.to_csv(export_path + "mc4")
    print(f"Curve-fitted {df.shape[0]} series, with {len(fit_models)} fit models > {elapsed(start_time)}")

    tcpl_write_data(dat=df, lvl=4, verbose=False)
    print(f"Stored L4 AEID {aeid} with {df.shape[0]} rows to db >> {elapsed(start_time)}")
    print("Done mc4")
    return df


def mc5(df):
    start_time = starting(f"mc5 with id {aeid}")

    if df is None:
        df = tcpl_load_data(lvl=4, fld='aeid', val=aeid, verbose=False)
        print(f"Loaded L4 AEID {aeid} with ({df.shape[0]} rows) >> {elapsed(start_time)}")

    assay_cutoff_methods = tcpl_mthd_load(lvl=5, aeid=aeid)["mthd"]
    bmad = df["bmad"].iloc[0]
    cutoffs = [mc5_mthds(mthd, bmad) for mthd in assay_cutoff_methods]
    cutoff = np.max(cutoffs) if len(cutoffs) > 0 else 0

    dat = get_mc5_data(aeid)
    dat = tcpl_hit(dat, cutoff, parallelize, verbose=verbose)

    print(f"Computed L5 AEID {aeid} hitcall parameters with {dat.shape[0]} rows >> {elapsed(start_time)}")
    tcpl_write_data(dat=dat, lvl=5, verbose=False)
    print(f"Stored L5 AEID {aeid} with {df.shape[0]} rows to db >> {elapsed(start_time)}")
    print("Done mc5")


def pipeline():
    start_time = starting("pipeline")
    # drop_tables(new_table_names) # uncomment if you want to remove the specified new tables from the db
    ensure_all_new_db_tables_exist()
    df = mc4() if do_fit else pd.read_csv(export_path + "mc4")
    mc5(df)
    export()
    print(f"Pipeline done >> {elapsed(start_time)}")


def export():
    start_time = starting(f"Export {aeid}")
    # Load the example level 5 data
    d1 = tcpl_load_data(lvl=5, fld="aeid", val=aeid, verbose=verbose)
    d1 = tcpl_prep_otpt(d1)

    # Here the consensus hit-call is 1 (active), and the fit categories are
    # all equal. Therefore, if the flags are ignored, the selected sample will
    # be the sample with the lowest modl_ga.
    df = tcpl_subset_chid(dat=d1, flag=False)  # [ , list(m4id, modl_ga)]
    if ml:
        df.to_csv(export_path + "chem.csv", header=True, index=True)
        # df = pd.read_csv(export_path+"chem.csv")

    print(elapsed(start_time))
    print("Done export")


if __name__ == '__main__':
    # Type `snakeviz pytcpl/profile/pipeline.prof` in terminal after run to view profile in browser
    if profile:
        name = "pipeline"
        with cProfile.Profile() as pr:
            pipeline()

        pr.dump_stats(f'profile/{name}.prof')
        print("Profiling complete")

    else:
        pipeline()
