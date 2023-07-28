import cProfile
import json
import os

from mc4_mthds import mc4_mthds
from pipeline_helper import starting, elapsed, ensure_all_new_db_tables_exist, load_config, \
    export_data, drop_tables, ROOT_DIR, track_fitted_params, get_assay_info
from pytcpl.pipeline_helper import store_cutoff
from tcpl_fit import tcpl_fit
from tcpl_hit import tcpl_hit
from tcpl_load_data import tcpl_load_data
from tcpl_mthd_load import tcpl_mthd_load
from tcpl_output import tcpl_output
from tcpl_write_data import tcpl_append

config = load_config()["pytcpl"]

import warnings
# warnings.filterwarnings("ignore")
# warnings.filterwarnings("error", category=RuntimeWarning)


def pipeline():
    aeid = config['aeid']
    assay_component_endpoint_name = get_assay_info(aeid)["assay_component_endpoint_name"]
    start_time = starting(f"Starting pipeline with assay id {aeid} ({assay_component_endpoint_name})")
    drop_tables(config["new_table_names"])  # Uncomment if you want to remove the specified pipeline tables from the db
    ensure_all_new_db_tables_exist()


    print(f"Load L3 data..")
    df = tcpl_load_data(lvl=3, fld='aeid', ids=aeid)
    get_bmad = tcpl_mthd_load(lvl=4, aeid=aeid)
    for mthd in list(get_bmad['mthd'].values): #+['onesd.aeid.lowconc.twells']
        df = mc4_mthds(mthd)(df)
    cutoff = store_cutoff(aeid, df)
    print(f"Loaded L3 with ({df.shape[0]} rows) >> {elapsed(start_time)}")

    print(f"Fit curves..")
    df = tcpl_fit(dat=df, fit_models=config["fit_models"], fit_strategy=config["fit_strategy"], cutoff=cutoff,
                  bidirectional=config["bidirectional"], parallelize=config["parallelize"], n_jobs=config["n_jobs"],
                  test=config["test"])
    track_fitted_params()
    print(f"Curve-fitted {df.shape[0]} series, with {len(config['fit_models'])} fit models > {elapsed(start_time)}")

    dat = tcpl_hit(df, config["fit_strategy"], cutoff, parallelize=config["parallelize"], n_jobs=config["n_jobs"])
    print(f"Computed hitcall parameters >> {elapsed(start_time)}")

    for col in ['concentration_unlogged', 'response', 'fitparams']:
        dat.loc[:, col] = dat[col].apply(json.dumps)
    tcpl_append(dat[config["output_cols"]], "output")
    print(f"Stored L5 with {dat.shape[0]} rows to db >> {elapsed(start_time)}")

    dat = dat[config['export_cols']]
    dat = tcpl_output(dat, aeid)
    dat = dat.rename(columns={"dsstox_substance_id": "dtxsid"})
    df_export = dat[['dtxsid', "chit"]]
    export_data(df_export, path=config["export_path"], folder="out", id=aeid)
    print(f"Done export >> {elapsed(start_time)}\n")

    print(f"Pipeline done >> {elapsed(start_time)}\n")


if __name__ == '__main__':
    if config["profile"]:
        with cProfile.Profile() as pr:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.filterwarnings("error")  # Raise warnings as exceptions
                pipeline()
        pr.dump_stats(os.path.join(ROOT_DIR, f'profile/pipeline.prof') )
        print("Profiling complete")  # Use `snakeviz pytcpl/profile/pipeline.prof` to view profile in browser
    else:
        pipeline()
