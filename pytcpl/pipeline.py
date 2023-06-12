import cProfile
import time
import warnings

import numpy as np
from matplotlib import pyplot as plt

from fit_models import get_fit_model
from mc4_mthds import mc4_mthds
from mc5_mthds import mc5_mthds
from query_db import tcpl_query
from tcpl_fit import tcpl_fit
from tcpl_hit import tcpl_hit
from tcpl_load_data import tcpl_load_data
from tcpl_mthd_load import tcpl_mthd_load
from tcpl_prep_otpt import tcpl_prep_otpt
from tcpl_subset_chid import tcpl_subset_chid
from tcpl_write_data import tcpl_write_data

# warnings.filterwarnings("ignore")
warnings.filterwarnings("error", category=RuntimeWarning)

aeid = 5
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
fit_models = ["cnst", "poly1", "poly2", "pow", "exp2", "exp3", "exp4", "exp5", "hill", "gnls"]
# fit_models = ["cnst", "poly1"]
# fit_models = ["cnst", "poly1", "poly2", "exp2"]


def pipeline():
    def mc4():
        start_time = starting(f"mc4 with id {aeid}")
        df = tcpl_load_data(lvl=3, fld='aeid', val=aeid)
        df = df.head(head) if test else df
        print(f"Loaded L3 AEID {aeid} with ({df.shape[0]} rows) >> {elapsed(start_time)}")

        get_bmad = tcpl_mthd_load(lvl=4, aeid=aeid)
        for mthd in get_bmad['mthd']:
            df = mc4_mthds(mthd)(df)

        mc4 = tcpl_fit(df, fit_models, bidirectional, force_fit=False, parallelize=parallelize, verbose=verbose)
        print(f"Curve-fitted {df.shape[0]} series, with {len(fit_models)} fit models > {elapsed(start_time)}")

        # track(mc4)

        tcpl_write_data(dat=mc4, lvl=4, verbose=False)
        print(f"Stored L4 AEID {aeid} with {df.shape[0]} rows to db >> {elapsed(start_time)}")
        print("Done mc4.")
        return mc4

    def track(mc4):
        pars_tracker = {key: [] for key in fit_models}
        modl_tracker = {key: [] for key in fit_models}
        conc_tracker = []
        resp_tracker = []
        fitparams = mc4["fitparams"]
        for i in range(fitparams.shape[0]):
            data = fitparams.iloc[i]
            mc4_row = mc4.iloc[i]
            conc_tracker.append(mc4_row["concentration_unlogged"])
            resp_tracker.append(mc4_row["response"])

            for model in list(data.keys()):
                params = data[model]
                pars = list(params["pars"].values())
                modl = params["modl"]
                pars_tracker[model].append(pars)
                modl_tracker[model].append(modl)

        # Define a qualitative colormap
        colormap = plt.cm.get_cmap('tab10')
        plt.figure()
        for i in range(len(modl_tracker["cnst"])):
            conc = np.array(conc_tracker[i])
            resp = np.array(resp_tracker[i])
            # uconc = np.unique(conc)
            # rmds = np.array([np.median(resp[conc == c]) for c in uconc])
            plt.plot(conc, resp, '-ok')
            for j, model in enumerate(fit_models):
                color = colormap(j)
                pars = pars_tracker[model][i]
                modl = modl_tracker[model][i]
                if modl:
                    x = np.linspace(np.min(conc), np.max(conc), 100)
                    pars = np.array(pars)
                    y = get_fit_model(model)(pars, x)
                    plt.plot(x, y, '-k', color=color, label=model)

        plt.legend(loc="upper left")
        # plt.show()
        plt.savefig(f'{export_path} + plot.png')

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
        print("Done mc5.")

    # Pipeline
    df = None
    df = mc4()
    mc5(df)
    # export()
    print("Pipeline done.")


def starting(pipeline_step):
    print(f"Starting {pipeline_step} ...")
    return time.time()


def elapsed(start_time):
    return f"Execution time in seconds: {str(round(time.time() - start_time, 2))}"


def get_mc5_data(aeid):
    mc4_name = "mc4_"
    mc4_param_name = "mc4_param_"
    query = f"SELECT {mc4_name}.m4id," \
            f"{mc4_name}.aeid," \
            f"{mc4_name}.logc_max," \
            f"{mc4_name}.logc_min," \
            f"{mc4_param_name}.model," \
            f"{mc4_param_name}.model_param," \
            f"{mc4_param_name}.model_val " \
            f"FROM {mc4_name} " \
            f"JOIN {mc4_param_name} " \
            f"ON {mc4_name}.m4id = {mc4_param_name}.m4id " \
            f"WHERE {mc4_name}.aeid = {aeid};"

    dat = tcpl_query(query)
    return dat


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
    print("Done export.")


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
