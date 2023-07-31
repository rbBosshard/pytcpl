import numpy as np

from query_db import query_db
from tcpl_fit_helper import mad


def mc4_mthds(mthd):
    return {
        'bmad.aeid.lowconc.twells': lambda df: df.assign(
            bmad=mad(df.loc[df['cndx'].isin([1, 2]) & (df['wllt'] == 't'), 'resp'])),
        'bmad.aeid.lowconc.nwells': lambda df: df.assign(bmad=mad(df.loc[df['wllt'] == 'n', 'resp'])),
        'onesd.aeid.lowconc.twells': lambda df: df.assign(
            osd=df.loc[df['cndx'].isin([1, 2]) & (df['wllt'] == 't'), 'resp'].std()),
        'bidirectional.false': lambda df: df.assign(bidirectional=False),
        'bmed.aeid.lowconc.twells': lambda df: df.assign(
            bmed=df.loc[df['cndx'].isin([1, 2]) & (df['wllt'] == 't'), 'resp'].median()),
        'no.gnls.fit': lambda df: df.assign(
            fit_models=[['cnst', 'hill', 'poly1', 'poly2', 'pow', 'exp2', 'exp3', 'exp4', 'exp5']]),
        'nmad.apid.null.zwells': lambda df: df.assign(
            bmad=df.groupby(['aeid', 'apid'])['resp'].apply(lambda x: mad(x))),
    }.get(mthd)


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
    flds = ["aeid", f"b.mc{lvl}_mthd AS mthd", f"b.mc{lvl}_mthd_id AS mthd_id"]
    tbls = [f"mc{lvl}_aeid AS a", f"mc{lvl}_methods AS b"]
    qstring = f"SELECT {', '.join(flds)} " \
              f"FROM {', '.join(tbls)} " \
              f"WHERE a.mc{lvl}_mthd_id = b.mc{lvl}_mthd_id " \
              f"AND aeid IN ({aeid});"
    return query_db(query=qstring)
