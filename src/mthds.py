import numpy as np

from query_db import query_db


def mc4_mthds(mthd, df):
    cndx = df['cndx'].isin([1, 2])
    wllt_t = df['wllt'] == 't'
    mask = df.loc[cndx & wllt_t, 'resp']

    if mthd == 'bmad.aeid.lowconc.twells':
        return mad(mask)
    elif mthd == 'bmad.aeid.lowconc.nwells':
        return mad(df.loc[df['wllt'] == 'n', 'resp'])
    elif mthd == 'onesd.aeid.lowconc.twells':
        return mask.std()
    elif mthd == 'bmed.aeid.lowconc.twells':
        return mask.median()


def mc5_mthds(mthd, bmad):
    return {
        'pc20': 20,
        'pc50': 50,
        'pc70': 70,
        'log2_1.2': np.log2(1.2),
        'log10_1.2': np.log10(1.2),
        'log2_2': np.log2(2),
        'log10_2': np.log10(2),
        'neglog2_0.88': -1 * np.log2(0.88),
        'coff_2.32': 2.32,
        'fc0.2': 0.2,
        'fc0.3': 0.3,
        'fc0.5': 0.5,
        'pc05': 5,
        'pc10': 10,
        'pc25': 25,
        'pc30': 30,
        'pc95': 95,
        'bmad1': bmad,
        'bmad2': bmad * 2,
        'bmad3': bmad * 3,
        'bmad4': bmad * 4,
        'bmad5': bmad * 5,
        'bmad6': bmad * 6,
        'bmad10': bmad * 10,
        # 'maxmed20pct': lambda df: df['max_med'].aggregate(lambda x: np.max(x) * 0.20),  # is never used
    }.get(mthd)


def tcpl_mthd_load(lvl, aeid):
    flds = [f"b.mc{lvl}_mthd AS mthd"]
    tbls = [f"mc{lvl}_aeid AS a", f"mc{lvl}_methods AS b"]
    qstring = f"SELECT {', '.join(flds)} " \
              f"FROM {', '.join(tbls)} " \
              f"WHERE a.mc{lvl}_mthd_id = b.mc{lvl}_mthd_id " \
              f"AND aeid IN ({aeid});"
    return query_db(query=qstring)["mthd"].tolist()


BMAD_CONSTANT = 1.4826


def mad(x):
    """Calculate the median absolute deviation (MAD) of an array"""
    return BMAD_CONSTANT * np.median(np.abs(x - np.median(x)))
