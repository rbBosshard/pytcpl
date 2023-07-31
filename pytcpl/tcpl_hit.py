import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from pipeline_helper import status, custom_format, get_msg_with_elapsed_time
from pytcpl.processing import tcpl_hit_core


def tcpl_hit(df, cutoff, config):
    total = df.shape[0]
    desc = get_msg_with_elapsed_time(f"{status('test_tube')}    - Third run (hit-call):   ", color_only_time=False)
    iterator = tqdm(df.iterrows(), desc=desc, total=total, bar_format=custom_format)

    if config['parallelize']:
        res = pd.DataFrame(Parallel(n_jobs=config['n_jobs'])(
            delayed(tcpl_hit_core)(
                params=row.fitparams,
                conc=np.array(row.concentration_unlogged),
                resp=np.array(row.response),
                cutoff=cutoff
            ) for _, row in iterator
        ))
    else:
        res = df.progress(lambda row: tcpl_hit_core(params=row.fitparams, conc=np.array(row.concentration_unlogged),
                                                    resp=np.array(row.response), cutoff=cutoff), axis=1, result_type='expand')

    df[res.columns] = res
    return df


