import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tcpl_hit_core import tcpl_hit_core


def tcpl_hit(mc4, fit_strategy, coff, parallelize=False, n_jobs=-1):
    if parallelize:
        res = pd.DataFrame(Parallel(n_jobs=n_jobs)(
            delayed(tcpl_hit_core)(
                fit_strategy=fit_strategy,
                params=row.fitparams,
                conc=np.array(row.concentration_unlogged),
                resp=np.array(row.response),
                cutoff=coff
            ) for _, row in mc4.iterrows()
        ))
    else:
        res = mc4.apply(lambda row: tcpl_hit_core(fit_strategy=fit_strategy, params=row.fitparams,
                              conc=np.array(row.concentration_unlogged),
                              resp=np.array(row.response), cutoff=coff), axis=1, result_type='expand')

    mc4[res.columns] = res
    return mc4
