import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.stats import t, chi2
from tqdm import tqdm
import warnings
import traceback

from src.pipeline.pipeline_constants import custom_format
from src.pipeline.models.models import get_model
from src.pipeline.models.objective_function import get_negative_log_likelihood
from src.pipeline.pipeline_helper import get_msg_with_elapsed_time, get_cutoff, init_aeid, load_method
from src.pipeline.pipeline_methods import mc6_mthds
from src.pipeline.models.track_fitted_params import track_fitted_params


def process(assay_endpoint_info, df, config, logger):
    """
    Process concentration-response data.

    This function processes concentration-response data stored in a DataFrame. It performs various steps such as
    grouping datapoints, curve fitting, and hit calling for each concentration-response series.

    Args:
        assay_endpoint_info (pandas.DataFrame): The DataFrame containing assay endpoint information.
        df (pandas.DataFrame): The DataFrame containing concentration-response data.
        config (dict): Configuration settings for data processing.
        logger: Logger object for logging messages.

    Returns:
        pandas.DataFrame: A DataFrame with processed concentration-response data.
    """
    aeid = assay_endpoint_info['aeid'].iloc[0]
    signal_direction = assay_endpoint_info['signal_direction'].iloc[0]
    assay_function_type = assay_endpoint_info['assay_function_type'].iloc[0]

    cutoffs = get_cutoff()
    cutoff = cutoffs.iloc[0]['cutoff']
    onesd = cutoffs.iloc[0]['onesd']
    bmad = cutoffs.iloc[0]['bmad']

    def group_datapoints_to_series(groups):
        """
        Group datapoints into concentration-response series.

        Args:
            groups (pandas.GroupBy): Grouped data by assay endpoint and substance.

        Returns:
            pandas.DataFrame: DataFrame with grouped concentration-response series.
        """

        def shrink_series(series):
            """
            Shrink a concentration-response series if it contains too many datapoints.

            This function takes a concentration-response series and checks if it contains too many datapoints to handle,
            which is often the case for positive control chemicals. If the series exceeds the specified threshold,
            it shrinks the series by calculating the median response for each unique concentration.

            Args:
                series (pandas.Series): A series containing concentration and response data.

            Returns:
                tuple: A tuple containing two lists - the shrunk concentrations and corresponding median responses,
                       or the original concentration and response if not exceeded.
            """
            conc, resp = series.conc, series.resp
            uconc = np.unique(conc)
            exceeded = len(conc) > config['max_num_datapoints_per_series_threshold']
            return (list(uconc), [pd.Series(resp)[conc == c].median() for c in uconc]) if exceeded else (conc, resp)

        df = groups.agg(conc=('logc', lambda x: list(10 ** x)), resp=('resp', list)).reset_index()
        df = df[df.conc.apply(lambda x: not any(pd.isna(x)))]
        df = df[:config['enable_data_subsetting']] if config['enable_data_subsetting'] else df
        df['conc'], df['resp'] = zip(*df.apply(shrink_series, axis=1))
        return df

    def check_to_fit(series):
        """
        Check if a series is eligible for curve fitting.

        Args:
            series (pandas.Series): A series containing concentration-response data.

        Returns:
            bool: True if the series is eligible for curve fitting, False otherwise.
        """
        conc, resp = np.array(series.conc), np.array(series.resp)
        rmds = np.array([np.median(resp[conc == c]) for c in np.unique(conc)])
        enough_concentrations_tested = rmds.size >= config['min_num_median_responses_threshold']
        high_responses_exist = np.any(abs(rmds) >= 0.8 * cutoff)
        enough_concentrations_tested_but_responses_to_low = enough_concentrations_tested and not high_responses_exist
        enough_concentrations_tested_and_high_responses_exist = enough_concentrations_tested and high_responses_exist
        return (not enough_concentrations_tested, enough_concentrations_tested_but_responses_to_low, enough_concentrations_tested_and_high_responses_exist)

    def fit(series):
        """
        Fit concentration-response data to curve models.

        Args:
            series (pandas.Series): A series containing concentration-response data.

        Returns:
            dict: Dictionary containing fit results for various curve models.
        """
        out = {}
        for fit_model in config['curve_fit_models']:
            conc = np.array(series.conc)
            resp = np.array(series.resp)
            signal_direction_is_bidirectional = signal_direction == 'bidirectional'
            x0 = get_model(fit_model)('x0')(signal_direction_is_bidirectional, conc, resp)
            bounds = get_model(fit_model)('bounds_bidirectional')(conc, resp) if signal_direction_is_bidirectional else get_model(fit_model)('bounds')(conc, resp)
            args = (conc, resp, get_model(fit_model)('fun'))

            fit_result = minimize(get_negative_log_likelihood, x0=x0, bounds=bounds, args=args,
                                  method=config['minimize_method'], tol=float(config['minimize_tol']),
                                  options={'maxiter': int(config['minimize_maxiter']),
                                           'maxfun': int(config['minimize_maxfun'])})

            pars = {k: v for k, v in zip(get_model(fit_model)('params'), fit_result.x)}
            ll = -fit_result.fun
            aic = 2 * len(pars) - 2 * ll
            out[fit_model] = {'pars': pars, 'll': ll, 'aic': aic}
        return out

    def hit(series):
        """
        Perform hit calling on a series.

        Args:
            series (pandas.Series): A series containing concentration-response data and fit results.

        Returns:
            dict: Dictionary containing hit call results for the series.
        """

        best_aic_model, hitcall, top, ac50, ac95, acc, actop, bmd, ac1sd, rmse, fitc = \
            None, None, None, None, None, None, None, None, None, None, None
        conc, resp, params = np.array(series.conc), np.array(series.resp), series.fit_params

        if params is None:
            hitcall = 0
        else:
            aics = {fit_model: params[fit_model]['aic'] for fit_model in params.keys()}

            if len(aics) == 0:
                return {"best_aic_model": "none"}

            if 'cnst' in aics and len(aics) == 1:
                return {"best_aic_model": "cnst"}

            best_aic_model = min({m: aics[m] for m in aics if m != "cnst"}, key=lambda k: aics[k])
            rel_likelihood = np.exp((aics[best_aic_model] - aics["cnst"]) / 2) if aics[best_aic_model] <= aics[
                "cnst"] else 1
            ps = params[best_aic_model]['pars']
            ps_list = list(ps.values())
            ll = params[best_aic_model]['ll']
            pred = get_model(best_aic_model)('fun')(conc, *ps_list[:-1]).tolist()
            rmse = np.sqrt(np.mean((resp - pred) ** 2))
            bmr = onesd * config['bmr_scale']  # bmr_scale is default 1.349
            top = pred[np.argmax(np.abs(pred))]  # top is taken to be highest model value

            inverse_function = get_model(best_aic_model)('inv')
            ac50 = inverse_function(.5 * top, *ps_list[:-1], conc)
            ac95 = inverse_function(.95 * top, *ps_list[:-1], conc)
            actop = inverse_function(top - np.sign(top) * 1e-12, *ps_list[:-1], conc)
            acc = inverse_function(np.sign(top) * cutoff, *ps_list[:-1], conc) if cutoff <= abs(top) else None
            ac1sd = inverse_function(np.sign(top) * onesd, *ps_list[:-1], conc) if onesd <= abs(top) else None
            bmd = inverse_function(np.sign(top) * bmr, *ps_list[:-1], conc) if bmr <= abs(top) else None

            # with warnings.catch_warnings(record=True) as w:
            #     warnings.filterwarnings("error", category=RuntimeWarning)
            #     try:
            #         ac50 = inverse_function(.5 * top, *ps_list[:-1], conc)
            #     except RuntimeWarning as rw:
            #         traceback_info = traceback.format_exc()
            #         print(top, ps_list[:-1])
            #         print(traceback_info)
            #         print("Caught a RuntimeWarning:", rw)
            #     except Exception as e:
            #         print("Caught an exception:", e)

            # Hitcall
            # Each p_i represents the odds of the curve being a hit according to different criteria;
            p1 = 1 - rel_likelihood  # p1 represents probability that constant model is correct

            p2 = 1
            unique_conc = np.unique(conc)
            for c in unique_conc:
                # multiply odds of each point falling below cutoff to get odds of all falling below,
                # use lower tail for positive top and upper tail for neg top
                median_resp = np.median(resp[conc == c])
                p = t.cdf(x=median_resp, loc=np.sign(top) * cutoff, scale=np.exp(ps['er']), df=4)
                p2 *= p if top < 0 else 1 - p
            p2 = 1 - p2  # odds of at least one point above cutoff

            # p3: get loglikelihood of top exactly at cutoff, use likelihood profile test
            # to calculate probability of being above cutoff

            ps_list[0] = get_model(best_aic_model)('scale')(cutoff, conc, ps_list)
            ll_cutoff = -get_negative_log_likelihood(params=ps_list, conc=conc, resp=resp,
                                                     fit_model=get_model(best_aic_model)(
                                                         'fun'))  # get log-likelihood at cutoff

            if abs(top) >= cutoff:
                p3 = (1 + chi2.cdf(2 * (ll - ll_cutoff), 1)) / 2
            else:
                p3 = (1 - chi2.cdf(2 * (ll - ll_cutoff), 1)) / 2

            hitcall = p1 * p2 * p3  # multiply three probabilities to get continuous hit odds overall

        # Compute fit category (fitc), see https://www.frontiersin.org/articles/10.3389/ftox.2023.1275980
        fitc = compute_fitc(ac50, ac95, conc, hitcall, top)
        if fitc is None:
            fitc = 2

        out = {'best_aic_model': best_aic_model, 'hitcall': hitcall, 'top': top, 'ac50': ac50, 'ac95': ac95, 'acc': acc,
               'actop': actop, 'bmd': bmd, 'ac1sd': ac1sd, 'fitc': int(fitc)}

        cautionary_flag_true = None
        if fitc >= 13:
            info = out.copy()
            info['cutoff'] = cutoff
            info['bmad'] = bmad
            info['resp'] = resp
            info['conc'] = conc
            info['rmse'] = rmse
            info['bidirectional'] = signal_direction == 'bidirectional'
            info['cell_viability_assay'] = assay_function_type.endswith('viability')
            cautionary_flags = [mc6_mthds(mthd, info) for mthd in load_method(lvl=6, aeid=aeid, config=config)]
            cautionary_flag_true = [key for flag_result in cautionary_flags for key, value in flag_result.items() if value]
            cautionary_flag_true = cautionary_flag_true if cautionary_flag_true else []
        out['cautionary_flags'] = [] if cautionary_flag_true is None else cautionary_flag_true

        return out

    def compute_fitc(ac50, ac95, conc, hitcall, top):
        cutoff_upper = 1.2 * cutoff
        cutoff_lower = 0.8 * cutoff
        min_conc, max_conc = np.min(conc), np.max(conc)

        fitc = 2
        if hitcall >= config['activity_threshold']:
            if abs(top) <= cutoff_upper:
                if ac50 <= min_conc:
                    fitc = 36
                elif ac50 > min_conc and ac95 < max_conc:
                    fitc = 37
                elif ac50 > min_conc and ac95 >= max_conc:
                    fitc = 38
            elif abs(top) > cutoff_upper:
                if ac50 <= min_conc:
                    fitc = 40
                elif ac50 > min_conc and ac95 < max_conc:
                    fitc = 41
                elif ac50 > min_conc and ac95 >= max_conc:
                    fitc = 42
        elif hitcall < config['activity_threshold']:
            if top and abs(top) < cutoff_lower:
                fitc = 13
            elif top and abs(top) >= cutoff_lower:
                fitc = 15
        else:
            fitc = 2

        return fitc

    # Preprocessing
    nj = config['n_jobs']
    p = nj != 1
    sample_groups = df.groupby(['aeid', 'spid'])
    logger.info(f"💻 Processing (fit ({signal_direction}) & hit) {len(sample_groups)} concentration-response series")
    df = group_datapoints_to_series(sample_groups)
    desc = get_msg_with_elapsed_time(f"🛠️  -> Preprocessing:  ")
    it = tqdm(df.iterrows(), total=df.shape[0], desc=desc, bar_format=custom_format)
    no_fit_nan, no_fit_too_low, to_fit = zip(*Parallel(n_jobs=nj)(delayed(check_to_fit)(i) for _, i in it) if p else [check_to_fit(i) for _, i in it])
    mask_no_fit_nan = pd.Series(no_fit_nan)
    mask_no_fit_too_low = pd.Series(no_fit_too_low)
    mask_to_fit = pd.Series(to_fit)

    df = df.reset_index()
    df_to_fit = df[mask_to_fit].reset_index(drop=True)
    df_no_fit_nan = df[mask_no_fit_nan].reset_index(drop=True)
    df_no_fit_too_low = df[mask_no_fit_too_low].reset_index(drop=True)

    # Curve-Fitting
    desc = get_msg_with_elapsed_time(f"☄️  -> Curve-Fitting: ")
    it = tqdm(df_to_fit.iterrows(), total=df_to_fit.shape[0], desc=desc, bar_format=custom_format)
    fit_params = Parallel(n_jobs=nj)(delayed(fit)(i) for _, i in it) if p else [fit(i) for _, i in it]
    df_fitted = df_to_fit.assign(fit_params=fit_params)
    df_no_fit_nan['fit_params'] = None
    df_no_fit_too_low['fit_params'] = None
    if config['enable_curve_fit_parameter_tracking']:
        track_fitted_params(df_fitted['fit_params'])

    # Hit-Calling
    df_no_fit_nan['hitcall'] = None
    df_no_fit_too_low['hitcall'] = 0
    df = pd.concat([df_fitted, df_no_fit_nan, df_no_fit_too_low]).reset_index(drop=True)
    desc = get_msg_with_elapsed_time(f"🚥  -> Hit-Calling:   ")
    it = tqdm(df.iterrows(), desc=desc, total=df.shape[0], bar_format=custom_format)
    res = pd.DataFrame(Parallel(n_jobs=nj)(delayed(hit)(i) for _, i in it)) if p \
        else df.apply(lambda i: hit(i), axis=1, result_type='expand')
    df[res.columns] = res

    # df = pd.concat([df_fitted, df_no_fit])
    for col in config['output_cols_filter']:
        if col not in df.columns:
            df[col] = None

    return df
