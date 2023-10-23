import numpy as np

from src.pipeline.models.helper import get_mad, get_mmed_conc, get_mmed


def mc6_mthds(mthd, info):
    """
    Compute specific warning flags based on input data for level 6 methods.
    """
    method_functions = {
        'ac50.lowconc': ac50_lowconc,
        'bmd.high': bmd_high,
        'border': border,
        'efficacy.50': efficacy_50,
        'gnls.lowconc': gnls_lowconc,
        'low.nconc': low_nconc,
        'low.nrep': low_nrep,
        'modl.directionality.fail': modl_directionality_fail,
        'multipoint.neg': multipoint_neg,
        'noise': noise,
        'singlept.hit.high': singlept_hit_high,
        'singlept.hit.mid': singlept_hit_mid,
        'viability.gnls': viability_gnls,
    }

    if mthd in method_functions:
        return method_functions[mthd](mthd, info)
    else:
        return None


def mc4_mthds(mthd, df):
    """
    Compute specific metrics based on input data for level 4 methods.

    Args:
        mthd (str): Method name.
        df (pandas.DataFrame): Input data.

    Returns:
        float: Computed metric value.
    """
    cndx = df['cndx'].isin([1, 2])
    wllt_t = df['wllt'] == 't'
    mask = df.loc[cndx & wllt_t, 'resp']

    if mthd == 'bmad.aeid.lowconc.twells':
        return get_mad(mask)
    elif mthd == 'bmad.aeid.lowconc.nwells':
        return get_mad(df.loc[df['wllt'] == 'n', 'resp'])
    elif mthd == 'onesd.aeid.lowconc.twells':
        return mask.std()
    elif mthd == 'bmed.aeid.lowconc.twells':
        return mask.median()


def mc5_mthds(mthd, bmad):
    """
    Get predefined values for level 5 methods or perform computations using bmad.

    Args:
        mthd (str): Method name.
        bmad (float): Calculated bmad value.

    Returns:
        float: Method-specific value.
    """
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


def ac50_lowconc(mthd, info):
    flag = "AC50 less than lowest concentration tested"

    flag_condition = (info["hitcall"] >= 0.9) and (info["ac50"] < np.min(info["conc"]))

    return {flag: flag_condition}


def modl_directionality_fail(mthd, info):
    flag = "Model directionality questionable"

    coffsign = -1 * info["cutoff"] if info["top"] < 0 else info["cutoff"]
    gtabscoff = abs(info["resp"]) > abs(coffsign)
    nrsp_gtabscoff = np.sum(gtabscoff)
    gtcoff = info["resp"] > coffsign
    ltcoff = info["resp"] < coffsign
    nrsp_gtcoff = np.sum(gtcoff)
    nrsp_ltcoff = np.sum(ltcoff)

    flag_condition = nrsp_gtabscoff > 2 * nrsp_gtcoff if coffsign > 0 else nrsp_gtabscoff > 2 * nrsp_ltcoff

    return {flag: flag_condition}


def bmd_high(mthd, info):
    flag = "Bmd > ac50, indication of high baseline variability"

    flag_condition = False
    if all(info[x] is not None for x in ["ac50", "bmd"]):
        flag_condition = info["bmd"] > info["ac50"]

    return {flag: flag_condition}


def border(mthd, info):
    flag = "Borderline"

    flag_condition = (abs(info["top"]) <= 1.2 * info["cutoff"]) & (abs(info["top"]) >= 0.8 * info["cutoff"])

    return {flag: flag_condition}


def efficacy_50(mthd, info):
    flag = "Less than 50% efficacy"

    flag_condition = (info["hitcall"] >= 0.9) and (abs(info["cutoff"]) >= 5) and \
                     ((abs(info["top"]) < 50) or (get_mmed(info['bidirectional'], info['conc'], info['resp']) < 50))

    flag_condition = flag_condition or (info["hitcall"] >= 0.9) and (abs(info["cutoff"]) < 5) and \
                     ((abs(info["top"]) < np.log2(1.5)) or (get_mmed(info['bidirectional'], info['conc'], info['resp']) < np.log2(1.5)))

    return {flag: flag_condition}


def gnls_lowconc(mthd, info):
    flag = "Gain AC50 < lowest conc when gnls or gnls2 is winning model"

    flag_condition = ((info["best_aic_model"] == "gnls") or (info["best_aic_model"] == "gnls2")) and \
                     (info["ac50"] < np.min(info["conc"]))

    return {flag: flag_condition}


def low_nconc(mthd, info):
    flag = "Number of concentrations tested is less than 4"

    nconc = len(np.unique(info['conc']))

    flag_condition = nconc <= 4

    return {flag: flag_condition}


def low_nrep(mthd, info):
    flag = "Average number of replicates per conc is less than 2"

    nrep = len(info['conc']) / len(np.unique(info['conc']))

    flag_condition = nrep < 2

    return {flag: flag_condition}


def multipoint_neg(mthd, info):
    flag = "Multiple points above baseline, inactive"

    nmed_gtbl = get_nmed_gtbl(info)
    flag_condition = (nmed_gtbl > 1) and (info["hitcall"] < 0.9)

    return {flag: flag_condition}


def get_nmed_gtbl(info):
    resp = np.array(info['resp'])
    conc = np.array(info['conc'])
    unique_conc = np.unique(conc)
    if info['bidirectional']:
        rmds = np.array([np.median(abs(resp[conc == c])) for c in unique_conc])
    else:
        rmds = np.array([np.median(resp[conc == c]) for c in unique_conc])
    med_rmds = rmds >= (3 * info['bmad'])
    nmed_gtbl = np.sum(med_rmds)
    return nmed_gtbl


def noise(mthd, info):
    flag = "Noisy data"

    flag_condition = info["rmse"] > info["cutoff"]

    return {flag: flag_condition}


def singlept_hit_high(mthd, info):
    flag = "Only highest conc above baseline, active"

    lstc = get_mmed_conc(info['bidirectional'], info['conc'], info['resp']) == np.max(info["conc"])

    nmed_gtbl = get_nmed_gtbl(info)

    flag_condition = (nmed_gtbl == 1) and (info["hitcall"] >= 0.9) and lstc

    return {flag: flag_condition}


def singlept_hit_mid(mthd, info):
    flag = "Only one conc above baseline, active"

    lstc = get_mmed_conc(info['bidirectional'], info['conc'], info['resp']) == np.max(info["conc"])

    nmed_gtbl = get_nmed_gtbl(info)

    flag_condition = (nmed_gtbl == 1) and (info["hitcall"] >= 0.9) and ~lstc

    return {flag: flag_condition}


def viability_gnls(mthd, info):
    flag = "Cell viability assay fit with gnls or gnls2 winning model"

    flag_condition = ((info["best_aic_model"] == "gnls") or (info["best_aic_model"] == "gnls2")) \
                     and (info["hitcall"] >= 0.9) & (info["cell_viability_assay"] == 1)

    return {flag: flag_condition}


