import time
import pandas as pd
from tcplFit2 import tcplFit2
from tcplLoadData import tcplLoadData
from tcplMthdLoad import tcplMthdLoad
from mc4_mthds import mc4_mthds
from tcplWriteData import tcplWriteData
from tcplPrepOtpt import tcplPrepOtpt
from tcplSubsetChid import tcplSubsetChid
from query_db import tcplQuery
from mc5_mthds import mc5_mthds
from tcplHit2 import tcplHit2

import warnings
warnings.filterwarnings("ignore")

id = 5
chunk_mc4 = 100
test = 0
verbose = 0

do_fit = 1
parallelize = 1
bidirectional = True
ml = 1

fitmodels = ["cnst", "poly1", "poly2" ]
# fitmodels = ["cnst", "hill", "gnls", "poly1", "poly2", "pow", "exp2", "exp3", "exp4", "exp5"]

## Pipeline
def prolog(pipeline_step):
    print(f"Starting {pipeline_step} ...")
    return time.time()

def print_elapsed_time(startTime):
    print('Execution time in seconds: ' + str(time.time() - startTime))


def mc4():
    startTime = prolog(f"mc4 with id {id}")
    
    if test:
        df = tcplLoadData(lvl=3, fld='aeid', val=id).head(chunk_mc4)
    else:
        df = tcplLoadData(lvl=3, fld='aeid', val=id)

    print_elapsed_time(startTime)
    
    ms = tcplMthdLoad(lvl = 4, id = id, type = "mc")
    if (ms.shape[0] == 0):
        print(f"No level 4 methods for AEID {id} Level 4 processing incomplete; no updates made to the mc4.")
        return
    
    print(f"Loaded L3 AEID {id} ({df.shape[0]} rows)")

    mthd_funcs = mc4_mthds()
    for method_key in ms['mthd']:
        df = mthd_funcs[method_key](df)

    
    if do_fit:
        df = tcplFit2(df, fitmodels, bidirectional, parallelize)
        df.to_csv("df.csv")
    else:
        df = pd.read_csv("df.csv")


    print(f"Curve-fitted {df.shape[0]} series")
    print_elapsed_time(startTime)

    tcplWriteData(dat = df, lvl = 4)
    print_elapsed_time(startTime)

    print("Done mc4.")


def mc5():
    startTime = prolog(f"mc5 with id {id}")

    df = tcplLoadData(lvl=4, fld='aeid', val=id)

    # Check if any level 4 data was loaded
    if df.shape[0] == 0:
        print(f"No level 4 data for AEID {id}. Level 5 processing incomplete; no updates made to the mc5 table for AEID {id}.")
        return

    print(f"Loaded L4 AEID {id} ({df.shape[0]} rows)")

    ms = tcplMthdLoad(lvl = 5, id = id, type = "mc")
    mthd_funcs = mc5_mthds(id)

    coff = []
    for method_key in ms['mthd']:
        coff.append(mthd_funcs[method_key](df))

    if ms.shape[0] == 0:
       print(f"No level 5 methods for AEID {id} -- cutoff will be 0.")

    # Determine final cutoff
    max_coff = max(coff)
    df['coff'] = max_coff[0]

    cutoff = max(df['coff'])

    mc4_name = "mc4_"
    mc5_name = "mc5_"
    mc4_param_name = "mc4_param_"

    query = f"SELECT {mc4_name}.m4id, {mc4_name}.aeid, {mc4_name}.spid, {mc4_name}.bmad, {mc4_name}.resp_max, {mc4_name}.resp_min, {mc4_name}.max_mean, {mc4_name}.max_mean_conc, {mc4_name}.max_med, {mc4_name}.max_med_conc, {mc4_name}.logc_max, {mc4_name}.logc_min, {mc4_name}.nconc, {mc4_name}.npts, {mc4_name}.nrep, {mc4_name}.nmed_gtbl, {mc4_name}.tmpi, {mc4_param_name}.model, {mc4_param_name}.model_param, {mc4_param_name}.model_val FROM {mc4_name} INNER JOIN {mc4_param_name} ON {mc4_name}.m4id = {mc4_param_name}.m4id WHERE {mc4_name}.aeid = {id};"
    dat = tcplQuery(query)

    dat = tcplHit2(dat, coff=cutoff)
    print_elapsed_time(startTime)

    tcplWriteData(dat = dat, lvl = 5)
    print_elapsed_time(startTime)

    print("Done mc5.")


def export():
    startTime = prolog(f"Export {id}")
    #get chemicals 
    #mc5 aeid -> spid -> chid -> casn/chnm/dsstox_substance_id
    ## Load the example level 5 data
    d1 = tcplLoadData(lvl = 5, fld = "aeid", val = id)
    d1 = tcplPrepOtpt(d1)
    
    ## Subset to an example of a duplicated chid
    # d2 <- d1[chid == 20182]
    # d2[ , list(m4id, hitc, fitc, modl_ga)]
    
    ## Here the consensus hit-call is 1 (active), and the fit categories are 
    ## all equal. Therefore, if the flags are ignored, the selected sample will
    ## be the sample with the lowest modl_ga.
    df = tcplSubsetChid(dat = d1, flag = False )#[ , list(m4id, modl_ga)]
    if ml:
        df.to_csv("chem.csv", header=True, index=True)
        # df = pd.read_csv("chem.csv")

    print_elapsed_time(startTime)
    print("Done export.")


# mc4()
# mc5()

if __name__ == '__main__':

    import cProfile
    name = "pipeline"

    with cProfile.Profile() as pr:
        export()
        

    pr.dump_stats(f'{name}.prof')

    print("Profiling complete")

    ## Type `snakeviz pipeline.prof ` in terminal to view profile in browser
