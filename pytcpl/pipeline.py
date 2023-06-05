import time
import pandas as pd
import cProfile
from tcplFit2 import tcpl_fit2
from tcplLoadData import tcpl_load_data
from tcplMthdLoad import tcpl_mthd_load
from mc4_mthds import mc4_mthds
from tcplWriteData import tcpl_write_data
from tcplPrepOtpt import tcpl_prep_otpt
from tcplSubsetChid import tcpl_subset_chid
from query_db import tcpl_query
from mc5_mthds import mc5_mthds
from tcplHit2 import tcpl_hit2


import warnings
warnings.filterwarnings("ignore")

aeid = 5
chunk_mc4 = 100
test = 0
verbose = 0
profile = 0

do_fit = 1
parallelize = 0
bidirectional = True
ml = 1

export_path = "export/"

fitmodels = ["cnst", "poly1", "poly2"]


# fitmodels = ["cnst", "hill", "gnls", "poly1", "poly2", "pow", "exp2", "exp3", "exp4", "exp5"]


# Pipeline
def prolog(pipeline_step):
    print(f"Starting {pipeline_step} ...")
    return time.time()


def print_elapsed_time(start_time):
    print('Execution time in seconds: ' + str(time.time() - start_time))


def mc4():
    start_time = prolog(f"mc4 with id {aeid}")

    if test:
        df = tcpl_load_data(lvl=3, fld='aeid', val=aeid).head(chunk_mc4)
    else:
        df = tcpl_load_data(lvl=3, fld='aeid', val=aeid)

    print_elapsed_time(start_time)

    ms = tcpl_mthd_load(lvl=4, id=aeid, type="mc")
    if ms.shape[0] == 0:
        print(f"No level 4 methods for AEID {aeid} Level 4 processing incomplete; no updates made to the mc4.")
        return

    print(f"Loaded L3 AEID {aeid} ({df.shape[0]} rows)")

    mthd_funcs = mc4_mthds()
    for method_key in ms['mthd']:
        df = mthd_funcs[method_key](df)

    if do_fit:
        df = tcpl_fit2(df, fitmodels, bidirectional, force_fit=False, parallelize=parallelize)
        df.to_csv(export_path+"df.csv")
    else:
        df = pd.read_csv(export_path+"df.csv")

    print(f"Curve-fitted {df.shape[0]} series")
    print_elapsed_time(start_time)

    tcpl_write_data(dat=df, lvl=4)
    print_elapsed_time(start_time)

    print("Done mc4.")


def mc5():
    start_time = prolog(f"mc5 with id {aeid}")

    df = tcpl_load_data(lvl=4, fld='aeid', val=aeid, verbose=False)

    # Check if any level 4 data was loaded
    if df.shape[0] == 0:
        print(
            f"No level 4 data for AEID {aeid}. "
            f"Level 5 processing incomplete; no updates made to the mc5 table for AEID {aeid}.")

        return

    print(f"Loaded L4 AEID {aeid} ({df.shape[0]} rows)")

    ms = tcpl_mthd_load(lvl=5, id=aeid, type="mc")
    mthd_funcs = mc5_mthds()

    coff = []
    for method_key in ms['mthd']:
        coff.append(mthd_funcs[method_key](df))

    if ms.shape[0] == 0:
        print(f"No level 5 methods for AEID {aeid} -- cutoff will be 0.")

    # Determine final cutoff
    max_coff = max(coff)
    df['coff'] = max_coff[0]

    cutoff = max(df['coff'])

    mc4_name = "mc4_"
    mc4_param_name = "mc4_param_"

    query = f"SELECT {mc4_name}.m4id," \
            f"{mc4_name}.aeid," \
            f"{mc4_name}.spid," \
            f"{mc4_name}.bmad," \
            f"{mc4_name}.resp_max," \
            f"{mc4_name}.resp_min," \
            f"{mc4_name}.max_mean," \
            f"{mc4_name}.max_mean_conc," \
            f"{mc4_name}.max_med," \
            f"{mc4_name}.max_med_conc," \
            f"{mc4_name}.logc_max," \
            f"{mc4_name}.logc_min," \
            f"{mc4_name}.nconc," \
            f"{mc4_name}.npts," \
            f"{mc4_name}.nrep," \
            f"{mc4_name}.nmed_gtbl," \
            f"{mc4_name}.tmpi," \
            f"{mc4_param_name}.model," \
            f"{mc4_param_name}.model_param," \
            f"{mc4_param_name}.model_val " \
            f"FROM {mc4_name} " \
            f"INNER JOIN {mc4_param_name} " \
            f"ON {mc4_name}.m4id = {mc4_param_name}.m4id " \
            f"WHERE {mc4_name}.aeid = {aeid};"

    dat = tcpl_query(query)

    dat = tcpl_hit2(dat, coff=cutoff, verbose=verbose)
    print_elapsed_time(start_time)

    tcpl_write_data(dat=dat, lvl=5)
    print_elapsed_time(start_time)

    print("Done mc5.")


def export():
    start_time = prolog(f"Export {aeid}")
    # Load the example level 5 data
    d1 = tcpl_load_data(lvl=5, fld="aeid", val=aeid, verbose=verbose)
    d1 = tcpl_prep_otpt(d1)

    # Here the consensus hit-call is 1 (active), and the fit categories are
    # all equal. Therefore, if the flags are ignored, the selected sample will
    # be the sample with the lowest modl_ga.
    df = tcpl_subset_chid(dat=d1, flag=False)  # [ , list(m4id, modl_ga)]
    if ml:
        df.to_csv(export_path+"chem.csv", header=True, index=True)
        # df = pd.read_csv(export_path+"chem.csv")

    print_elapsed_time(start_time)
    print("Done export.")


if __name__ == '__main__':

    if profile:
        name = "pipeline"
        with cProfile.Profile() as pr:
            export()

        pr.dump_stats(f'{name}.prof')
        print("Profiling complete")
        # Type `snakeviz pipeline.prof ` in terminal to view profile in browser

    else:
        mc4()
        mc5()
        export()
