import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.stats import t, chi2
from tqdm import tqdm

from src.pipeline.pipeline_constants import custom_format
from src.pipeline.models.models import get_model
from src.pipeline.models.objective_function import get_negative_log_likelihood
from src.pipeline.pipeline_helper import get_msg_with_elapsed_time, get_cutoff
from src.pipeline.models.track_fitted_params import track_fitted_params


def process(df, config, logger):
    """
    Process concentration-response data.

    This function processes concentration-response data stored in a DataFrame. It performs various steps such as
    grouping datapoints, curve fitting, and hit calling for each concentration-response series.

    Args:
        df (pandas.DataFrame): The DataFrame containing concentration-response data.
        config (dict): Configuration settings for data processing.
        logger: Logger object for logging messages.

    Returns:
        pandas.DataFrame: A DataFrame with processed concentration-response data.
    """
    cutoffs = get_cutoff()
    cutoff = cutoffs.iloc[0]['cutoff']
    onesd = cutoffs.iloc[0]['onesd']

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
        return (rmds.size >= config['min_num_median_responses_threshold']) and np.any(rmds >= cutoff)

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
            x0 = get_model(fit_model)('x0')(conc, resp)
            bounds = get_model(fit_model)('bounds')(conc, resp)
            args = (conc, resp, get_model(fit_model)('fun'))
            fit_result = minimize(get_negative_log_likelihood, x0=x0, bounds=bounds, args=args)
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
        conc, resp, params = np.array(series.conc), np.array(series.resp), series.fit_params
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
        # rmse = np.sqrt(np.mean((resp - pred) ** 2))
        top = np.max(np.abs(pred))  # top is taken to be highest model value
        inverse_function = get_model(best_aic_model)('inv')
        ac50 = inverse_function(.5 * top, *ps_list[:-1], conc)
        acc = inverse_function(cutoff, *ps_list[:-1], conc) if cutoff <= top else None
        actop = inverse_function(top, *ps_list[:-1], conc)
        ac1sd = inverse_function(onesd, *ps_list[:-1], conc)
        bmr = onesd * config['bmr_scale']  # bmr_scale is default 1.349
        bmd = inverse_function(bmr, *ps_list[:-1], conc)

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

        ps_list[0] = get_model(best_aic_model)('scale')(cutoff, conc, ps_list)
        ll_cutoff = -get_negative_log_likelihood(params=ps_list, conc=conc, resp=resp,
                                                 fit_model=get_model(best_aic_model)(
                                                     'fun'))  # get log-likelihood at cutoff
        p3 = (1 + np.sign(top) * chi2.cdf(2 * (ll - ll_cutoff), 1)) / 2

        continuous_hitcall = p1 * p2 * p3  # multiply three probabilities to get continuous hit odds overall

        out = {'best_aic_model': best_aic_model, 'hitcall': continuous_hitcall, 'top': top, 'ac50': ac50, 'acc': acc,
               'actop': actop, 'bmd': bmd, 'ac1sd': ac1sd}

        return out

    # Preprocessing
    nj = config['n_jobs']
    p = nj != 1
    sample_groups = df.groupby(['aeid', 'spid'])
    logger.info(f"💻 Processing (fit & hit) {len(sample_groups)} concentration-response series")
    df = group_datapoints_to_series(sample_groups)
    desc = get_msg_with_elapsed_time(f"🛠️  -> Preprocessing:  ")
    it = tqdm(df.iterrows(), total=df.shape[0], desc=desc, bar_format=custom_format)
    mask = Parallel(n_jobs=nj)(delayed(check_to_fit)(i) for _, i in it) if p else [check_to_fit(i) for _, i in it]
    mask = pd.Series(mask)
    df = df.reset_index()
    df_to_fit = df[mask].reset_index(drop=True)
    df_no_fit = df[~mask].reset_index(drop=True)

    # Curve-Fitting
    desc = get_msg_with_elapsed_time(f"☄️  -> Curve-Fitting: ")
    it = tqdm(df_to_fit.iterrows(), total=df_to_fit.shape[0], desc=desc, bar_format=custom_format)
    fit_params = Parallel(n_jobs=nj)(delayed(fit)(i) for _, i in it) if p else [fit(i) for _, i in it]
    df_fitted = df_to_fit.assign(fit_params=fit_params)
    df_no_fit['fit_params'] = None
    if config['enable_curve_fit_parameter_tracking']:
        track_fitted_params(df_fitted['fit_params'])

    # Hit-Calling
    df_fitted['hitcall'] = 0
    df_no_fit['hitcall'] = 0
    desc = get_msg_with_elapsed_time(f"🚥  -> Hit-Calling:   ")
    it = tqdm(df_fitted.iterrows(), desc=desc, total=df_fitted.shape[0], bar_format=custom_format)
    res = pd.DataFrame(Parallel(n_jobs=nj)(delayed(hit)(i) for _, i in it)) if p \
        else df_fitted.apply(lambda i: hit(i), axis=1, result_type='expand')
    df_fitted[res.columns] = res

    df = pd.concat([df_fitted, df_no_fit])
    for col in config['output_cols_filter']:
        if col not in df.columns:
            df[col] = None

    return df
