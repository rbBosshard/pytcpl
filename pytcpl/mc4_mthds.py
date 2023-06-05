from mad import mad


def mc4_mthds():
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
            fitmodels=[['cnst', 'hill', 'poly1', 'poly2', 'pow', 'exp2', 'exp3', 'exp4', 'exp5']]),
        'nmad.apid.null.zwells': lambda df: df.assign(
            bmad=df.groupby(['aeid', 'apid'])['resp'].apply(lambda x: mad(x))),
    }
