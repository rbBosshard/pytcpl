import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from fit_models import get_params
from tcpl_hit_core import tcpl_hit_core
from tcpl_load_data import tcpl_load_data


def tcpl_hit(mc4, fit_strategy, coff, parallelize=False, n_jobs=-1):
    if parallelize:
        test = Parallel(n_jobs=n_jobs)(
            delayed(tcpl_hit_core)(fit_strategy=fit_strategy,
                                   params=row.fitparams, conc=np.array(row.concentration_unlogged),
                                   resp=np.array(row.response), cutoff=coff) for _, row in mc4.iterrows()
        )
        res = pd.concat([mc4, pd.DataFrame(test)], axis=1)

    else:
        test = (
            mc4.assign(df=lambda row: [
                tcpl_hit_core(fit_strategy=fit_strategy, params=row.fitparams,
                              conc=np.array(row.concentration_unlogged),
                              resp=np.array(row.response), cutoff=coff)
                for _, row in row.iterrows()]).drop(['concentration_unlogged', 'response'], axis=1)
        )

        res = pd.concat([mc4, pd.DataFrame(test['df'].tolist())], axis=1)

    return res


def get_nested_mc4(mc4, fit_strategy, parallelize, n_jobs=-1):
    df = mc4[mc4['model'] != 'all']
    def tcpl_fit_nest(dat):
        modelnames = dat["model"].unique()
        dicts = {}
        for m in modelnames:
            df = dat[dat["model"] == m].groupby("model_param")["model_val"].apply(lambda x: float(x.iloc[0]))
            dicts[m] = df.to_dict()
            dicts[m]["pars"] = {x: dicts[m][x] for x in get_params(m, fit_strategy)}
        return dicts

    if parallelize:  # Parallel: Split the DataFrame into groups and apply the function in parallel
        def apply_tcpl_fit_nest(name, group):
            out = tcpl_fit_nest(group[['model', 'model_param', 'model_val']])
            return pd.DataFrame({"m4id": [name], "params": [out]})

        nested_mc4 = Parallel(n_jobs=n_jobs)(delayed(apply_tcpl_fit_nest)(name, group)
                                         for name, group in df.groupby('m4id'))
        nested_mc4 = pd.concat(nested_mc4).reset_index(drop=True)  # Concatenate the results into a single DataFrame

    else:  # Serial: For debugging
        nested_mc4 = df.groupby('m4id').apply(lambda x: tcpl_fit_nest(
            pd.DataFrame({'model': x['model'], 'model_param': x['model_param'],
                          'model_val': x['model_val']})))
        nested_mc4 = nested_mc4.reset_index(name='params')
    return nested_mc4
