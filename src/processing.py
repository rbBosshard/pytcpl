import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.stats import t, chi2
from tqdm import tqdm

from constants import custom_format, custom_format_
from fit_models import get_model
from fit_models import get_params
from pipeline_helper import track_fitted_params, get_msg_with_elapsed_time, status, print_
from src.fit_models import INITIAL_VALUES, BOUNDS
from tcpl_obj_fn import tcpl_obj


def processing(df, cutoff, config):
    print_(f"{status('laptop')} Processing {df.shape[0]} concentration-response series")
    df = preprocess(df, config)

    def all_models_fit(series):
        out = {}
        for model in config['curve_fit_models']:
            fit, ll = fit_curve(model, np.array(series['conc']), np.array(series['response']))
            pars = {k: v for k, v in zip(get_params(model), fit)}
            out.update({model: {'pars': pars, 'll': ll}})
        return out

    def preprocess_series(series):
        conc = np.array(series['conc'])
        resp = np.array(series['response'])
        rmds = np.array([np.median(resp[conc == c]) for c in np.unique(conc)])
        to_fit = (rmds.size >= 4) and np.any(np.abs(rmds) >= cutoff)
        fit, ll = fit_curve('cnst', conc, resp)
        pars = {key: value for key, value in zip(get_params('cnst'), fit)}
        out = {'cnst': {'pars': pars, 'll': ll}}
        return out, to_fit

    # Preprocessing & Filtering
    desc = get_msg_with_elapsed_time(f"{status('hammer_and_wrench')}  -> Preprocessing:  ", color_only_time=False)
    it = tqdm(df.iterrows(), total=df.shape[0], desc=desc, bar_format=custom_format)
    if config["n_jobs"] != 1:
        fit_params_cnst, fits = map(list, zip(*Parallel(n_jobs=config['n_jobs'])(delayed(preprocess_series)(i) for _, i in it)))
    else:
        fit_params_cnst, fits = map(list, zip(*(preprocess_series(row) for _, row in it)))

    # Curve-Fitting
    desc = get_msg_with_elapsed_time(f"{status('comet')}  -> Curve-Fitting: ", color_only_time=False)
    it = tqdm(df[fits].iterrows(), total=df[fits].shape[0], desc=desc, bar_format=custom_format_)
    if config["n_jobs"] != 1:
        fit_params = Parallel(n_jobs=config['n_jobs'])(delayed(all_models_fit)(i) for _, i in it)
    else:
        fit_params = [all_models_fit(i) for _, i in it]

    # Merge fit_params
    mask = np.array([{} for _ in range(len(fit_params_cnst))])
    mask[fits] = fit_params
    fit_params = [{**dict1, **dict2} for dict1, dict2 in zip(fit_params_cnst, mask)]
    df = df.assign(fit_params=fit_params)

    if config['enable_curve_fit_parameter_tracking']:
        track_fitted_params(df['fit_params'])

    # Hit-Calling
    desc = get_msg_with_elapsed_time(f"{status('horizontal_traffic_light')}  -> Hit-Calling:   ", color_only_time=False)
    it = tqdm(df.iterrows(), desc=desc, total=df.shape[0], bar_format=custom_format)

    if config["n_jobs"] != 1:
        res = pd.DataFrame(Parallel(n_jobs=config['n_jobs'])(
            delayed(tcpl_hit_core)(
                params=i.fit_params,
                conc=np.array(i.conc),
                resp=np.array(i.response),
                cutoff=cutoff
            ) for _, i in it
        ))
    else:
        res = df.apply(lambda i: tcpl_hit_core(params=i.fit_params, conc=np.array(i.conc), resp=np.array(i.response),
                                               cutoff=cutoff), axis=1, result_type='expand')
    df[res.columns] = res
    return df


def preprocess(df, config):
    df = df.groupby(['aeid', 'spid']).agg(conc=('logc', lambda x: list(10 ** x)), response=('resp', list)).reset_index()
    df = df[df.conc.apply(lambda x: not any(pd.isna(x)))]
    df = df[:config['enable_data_subsetting']] if config['enable_data_subsetting'] else df

    def shrink_series(row):
        conc = row['conc']
        resp = row['response']
        uconc = pd.unique(conc)
        t = len(conc) > config['num_datapoints_per_series_threshold']
        return (list(uconc), [pd.Series(resp)[conc == c].median() for c in uconc]) if t else (list(conc), list(resp))

    # Shrink series containing too many datapoints to handle. Often the case for positive control chemical
    df['conc'], df['response'] = zip(*df.apply(shrink_series, axis=1))

    return df


def tcpl_hit_core(params, conc, resp, cutoff, onesd=1, bmr_scale=1.349, bmed=0, bmd_low_bnd=None, bmd_up_bnd=None):
    aics = {}
    fit_models = params.keys()
    for fit_model in fit_models:
        ll = params[fit_model]['ll']
        aic = 2 * len(params[fit_model]['pars']) - 2 * ll
        if aic >= 0:
            aics.update({fit_model: aic})

    if len(aics) == 0:
        best_aic_model = "none"
        return {"best_aic_model": best_aic_model, "hitcall": 0}

    if 'cnst' in aics and len(aics) == 1:
        best_aic_model = "cnst"
        return {"best_aic_model": best_aic_model, "hitcall": 0}

    best_aic_model = min({m: aics[m] for m in aics if m != "cnst"}, key=lambda k: aics[k])
    rel_likelihood = np.exp((aics[best_aic_model] - aics["cnst"]) / 2) if aics[best_aic_model] <= aics["cnst"] else 1
    ps = params[best_aic_model]['pars']
    ps_list = list(ps.values())
    ll = params[best_aic_model]['ll']
    pred = get_model(best_aic_model)(conc, *ps_list[:-1]).tolist()
    rmse = np.sqrt(np.mean((resp - pred) ** 2))

    if best_aic_model in ("poly1", "poly2", "pow"):
        top = np.max(np.abs(pred))  # top is taken to be highest model value
        ac50 = get_model(best_aic_model + "_")(.5 * top, *ps_list[:-1])
    elif best_aic_model in ("hill", "exp4", "exp5"):
        top = ps["tp"]  # methods with a theoretical top/ac50
        ac50 = ps["ga"]
    elif best_aic_model == "gnls":
        top = get_model(best_aic_model + "_")(ps["tp"], *ps_list[:-1])
        ac50 = get_model(best_aic_model + "_")(.5 * top, *ps_list[:-1])
    else:
        raise NotImplementedError()

    # Each p_i represents the odds of the curve being a hit according to different criteria;
    p1 = 1 - rel_likelihood  # p1 represents probability that constant model is correct

    p2 = 1
    for y in np.array([np.median(resp[conc == c]) for c in np.unique(conc)]):
        # multiply odds of each point falling below cutoff to get odds of all falling below,
        # use lower tail for positive top and upper tail for neg top
        p = t.cdf(x=y, loc=np.sign(top) * cutoff, scale=np.exp(ps['er']), df=4)
        p2 *= p if top < 0 else 1 - p
    p2 = 1 - p2  # odds of at least one point above cutoff

    if best_aic_model in ["exp4", "exp5", "hill", "gnls"]:
        ps_list[0] = cutoff
    elif best_aic_model == "poly1":
        ps_list[0] = cutoff / np.max(conc)
    elif best_aic_model == "poly2":
        ps_list[0] = cutoff / (np.max(conc) / ps_list[1] + (np.max(conc) / ps_list[1]) ** 2)
    elif best_aic_model == "pow":
        ps_list[0] = cutoff / (np.max(conc) ** ps_list[1])

    ll_cutoff = -tcpl_obj(params=ps_list, conc=conc, resp=resp, fit_model=get_model(best_aic_model))  # get log-likelihood at coff
    p3 = (1 + np.sign(top) * chi2.cdf(2 * (ll - ll_cutoff), 1)) / 2

    hitcall = p1 * p2 * p3  # multiply three probabilities to get continuous hit odds overall

    # bmr = onesd * bmr_scale  # bmr_scale is default 1.349
    # if hitcall > 0:
    #     acc = acy(np.sign(top) * cutoff, pars | {"top": top}, fit_model=best_aic_model)
    #     ac1sd = acy(np.sign(top) * onesd, pars, fit_model=best_aic_model)
    #     bmd = acy(np.sign(top) * bmr, ps, fit_model=best_aic_model)
    var_list = ["fit_model", "hitcall", "ac50", "top"]  # + ["acc", "ac1sd", "bmd"]
    out = {var: value for var, value in locals().items() if var in var_list and value is not None}
    return out


def fit_curve(fit_model, conc, resp):
    initial_value_er = [] if fit_model == 'cnst' else [0.9]
    bounds_er = () if fit_model == 'cnst' else ((-1, 5),)
    initial_values = INITIAL_VALUES[fit_model] + initial_value_er
    bounds = BOUNDS[fit_model] + bounds_er
    args = (conc, resp, get_model(fit_model))
    fit = minimize(tcpl_obj, x0=initial_values, bounds=bounds, args=args)
    return fit.x, -fit.fun
