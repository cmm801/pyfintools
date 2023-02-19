""" Provides library of common time series functions and an extension to pandas Series/DataFrame objects.

    For supported time series types, you can convert between price 'levels' and 'returns and vice versa.
    Functions exist for which can convert the data frequency, and computing common time series statistics
    such as annual returns, cumulative returns, volatility, CVaR, VaR, Sharpe ratio, etc.

    To use the pandas extension, a 'ts_type' variable must be defined on the pandas object,
    and set to one of 'levels', 'returns', 'log_returns', 'rates' or 'dates'. Then, the extension

    Examples:
    If one starts with a variable 'ts', which is a pandas DataFrame object representing asset prices, 
    then the maximum draw-down can be computed as follows
    >> ts.ts_type = 'levels'
    >> ts.fts.max_drawdown()
    
    One could go even further and compute common summary statistics for the time series by running
    >> ts.fts.summary_stats()    
    
    One could then convert the data from prices ('levels') to monthly log returns by running
    >> ts.fts.to_log_returns(sampling_freq='M')

"""

import pandas as pd
import numpy as np

import secdb.tools.freq
import secdb.tools.stats


UNSMOOTH_METHOD_GELTNER = 'geltner'
UNSMOOTH_METHOD_GELTNER_ROLLING = 'geltner_rolling'
DEFAULT_SAMPLING_FREQUENCY = 'M'

SIMPLE_RETURN = 'simple'
LOG_RETURN = 'log'
DIFF_RETURN = 'diff'
ARITHMETIC_MEAN = 'arithmetic'
GEOMETRIC_MEAN = 'geometric'

TS_TYPE_LEVELS = 'levels'
TS_TYPE_SIMPLE_RETURNS = 'returns'
TS_TYPE_LOG_RETURNS = 'log_returns'
TS_TYPE_RATES = 'rates'
TS_TYPE_DATES = 'dates'
TS_TYPE_UNKNOWN = ''

NUMERIC_TS_TYPES = [TS_TYPE_LEVELS, TS_TYPE_SIMPLE_RETURNS, TS_TYPE_LOG_RETURNS, TS_TYPE_RATES]
VALID_TS_TYPES = NUMERIC_TS_TYPES + [TS_TYPE_DATES]


@pd.api.extensions.register_series_accessor("fts")
@pd.api.extensions.register_dataframe_accessor("fts")


class PerformanceAccessor(object):
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        # Check that this is a pandas time series object
        assert isinstance(obj.index, pd.DatetimeIndex), 'The object index must be of time pandas.DatetimeIndex.' 

    def _check_is_numeric(self, allow_unknown=False):
        if not hasattr(self._obj, 'ts_type'):
            raise AttributeError(f"Pandas object must have 'ts_type' equal to one of: {NUMERIC_TS_TYPES}")
        else:
            return check_valid_ts_type(self._obj.ts_type, allow_unknown=allow_unknown)

    @property
    def ts_type(self):
        self._check_is_numeric(allow_unknown=True)
        return self._obj.ts_type
    
    @property
    def frequency(self):
        return secdb.tools.freq.infer_freq(self._obj.index, allow_missing=True)
    
    @property
    def n_periods_per_year(self):
        return secdb.tools.freq.get_periods_per_year(self.frequency)
    
    @property
    def n_years(self):
        return get_num_years(self._obj)
    
    def convert_ts_type(self, target_type, sampling_freq=None):
        self._check_is_numeric()
        return convert_ts_type(self._obj, orig_type=self.ts_type, 
                               target_type=target_type, sampling_freq=sampling_freq)

    def to_levels(self, sampling_freq=None):
        return self.convert_ts_type(target_type=TS_TYPE_LEVELS, sampling_freq=sampling_freq)

    def to_simple_returns(self, sampling_freq=None):
        return self.convert_ts_type(target_type=TS_TYPE_SIMPLE_RETURNS, sampling_freq=sampling_freq)

    def to_log_returns(self, sampling_freq=None):
        return self.convert_ts_type(target_type=TS_TYPE_LOG_RETURNS, sampling_freq=sampling_freq)

    def summary_stats(self, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, skipna=True, bmk=None):
        self._check_is_numeric()
        if bmk is not None:
            bmk_ts = bmk.fts.convert_ts_type(target_type=self.ts_type, sampling_freq=self.frequency)
        else:
            bmk_ts = None
        return get_summary_stats(self._obj, stype=self.ts_type, sampling_freq=sampling_freq, 
                                 skipna=skipna, bmk=bmk_ts)

    def summary_stats_max_drawdown(self, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, skipna=True):
        self._check_is_numeric()
        return get_max_drawdown_summary_stats(self._obj, stype=self.ts_type, skipna=skipna)
    
    def volatility(self, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, use_log_rtns=True, skipna=True):
        self._check_is_numeric()
        return volatility(self._obj, stype=self.ts_type, sampling_freq=sampling_freq, 
                          use_log_rtns=use_log_rtns, skipna=skipna)
        
    def downside_risk(self, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, use_log_rtns=True, skipna=True):
        self._check_is_numeric()
        return downside_risk(self._obj, stype=self.ts_type, sampling_freq=sampling_freq, 
                             use_log_rtns=use_log_rtns, skipna=skipna)

    def upside_risk(self, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, use_log_rtns=True, skipna=True):
        self._check_is_numeric()
        return upside_risk(self._obj, stype=self.ts_type, sampling_freq=sampling_freq, 
                           use_log_rtns=use_log_rtns, skipna=skipna)
    
    def drawdown(self):
        self._check_is_numeric()
        return drawdown(self._obj, stype=self.ts_type)

    def max_drawdown(self):
        self._check_is_numeric()
        return max_drawdown(self._obj, stype=self.ts_type)

    def VaR(self, q, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, horizon='12M'):
        self._check_is_numeric()
        return VaR(self._obj, q=q, stype=self.ts_type, sampling_freq=sampling_freq, horizon=horizon)

    def CVaR(self, q, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, horizon='12M'):
        self._check_is_numeric()
        return CVaR(self._obj, q=q, stype=self.ts_type, sampling_freq=sampling_freq, horizon=horizon)

    def cum_return(self, skipna=True):
        self._check_is_numeric()
        return cum_return(self._obj, stype=self.ts_type, skipna=skipna)

    def ann_return(self, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, mean_type=GEOMETRIC_MEAN, skipna=True):
        self._check_is_numeric()
        return ann_return(self._obj, stype=self.ts_type, sampling_freq=sampling_freq, mean_type=mean_type, skipna=skipna)

    def first_valid_rows(self):
        return get_first_valid_rows(self._obj)
    
    def last_valid_rows(self):
        return get_last_valid_rows(self._obj)

    def first_valid_index(self):
        return get_first_valid_index(self._obj)

    def last_valid_index(self):
        return get_last_valid_index(self._obj)

    def num_observations(self):
        return get_num_observations(self._obj)
    
    def get_backfilled_ts(self, bf_ts, bf_type, bf_date=None, bf_fun=None):
        return get_backfilled_ts(self._obj, bf_ts=bf_ts, stype=self.ts_type, 
                                 bf_type=bf_type, bf_date=bf_date, bf_fun=bf_fun)
                
    def unsmooth_returns(self, method=UNSMOOTH_METHOD_GELTNER, **kwargs):
        """ Apply an unsmoothing method to the time series to remove autocorrelation. """
        return unsmooth_returns(self._obj, stype=self.ts_type, method=method, **kwargs)
    
    def fundamental_frequency(self):
        return get_fundamental_frequencies(self._obj, stype=self.ts_type)

    def convert_frequency(self, target_frequency):
        if isinstance(self._obj, pd.Series):
            ts_types = [self.ts_type]
        else:
            ts_types = [self.ts_type] * self._obj.shape[1]
        return convert_frequency(self._obj, ts_types=ts_types, target_frequency=target_frequency)



def get_summary_stats(ts, stype, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, skipna=True, bmk=None):
    """ Get a DataFrame with summary statistics for the input time series.
    """
    values = [cum_return(ts, stype=stype, skipna=skipna),
              ann_return(ts, stype=stype, sampling_freq=sampling_freq, mean_type=ARITHMETIC_MEAN, skipna=skipna),
              ann_return(ts, stype=stype, sampling_freq=sampling_freq, mean_type=GEOMETRIC_MEAN, skipna=skipna),
              volatility(ts, stype=stype, sampling_freq=sampling_freq, skipna=skipna),
              downside_risk(ts, stype=stype, sampling_freq=sampling_freq, skipna=skipna),
              upside_risk(ts, stype=stype, sampling_freq=sampling_freq, skipna=skipna),
              VaR(ts, q=0.05, stype=stype, sampling_freq=sampling_freq),
              CVaR(ts, q=0.05, stype=stype, sampling_freq=sampling_freq),              
              get_first_valid_index(ts),
              get_last_valid_index(ts),
              get_num_observations(ts),
             ]
    
    # Add statistics that depend on a benchmark, if one is provided
    if bmk is not None:
        values.extend([
                        information_ratio(ts, bmk, stype=stype, sampling_freq=sampling_freq),
                        tracking_error(ts, bmk, stype=stype, sampling_freq=sampling_freq),
                        ann_excess_return(ts, bmk, stype=stype, sampling_freq=sampling_freq, mean_type=ARITHMETIC_MEAN),
                        ann_excess_return(ts, bmk, stype=stype, sampling_freq=sampling_freq, mean_type=GEOMETRIC_MEAN),            
        ])

    # Create a pandas DataFrame from the statistics
    stats = pd.concat(values, axis=1)

    # Add the draw-down statistics
    mdd_stats = get_max_drawdown_summary_stats(ts, stype, skipna=skipna)
    mdd_stats.columns = 'mdd_' + mdd_stats.columns
    return pd.concat([stats, mdd_stats], axis=1)

def get_max_drawdown_summary_stats(ts, stype, skipna=True):
    # Make sure the series is a pandas DataFrame (not a pandas Series)
    if isinstance(ts, pd.Series):
        cols = [ts.name]
        ts = pd.DataFrame(ts)
        ts.columns = cols
        
    # Calculate the draw-down
    mdd_ts = drawdown(ts, stype)
    
    # Loop through the columns one by one and calculate the summary draw-down statistics
    results = []
    for col in mdd_ts.columns:
        sub_ts = mdd_ts[col]
        if not skipna:
            sub_ts = sub_ts.dropna()
        res = _get_max_drawdown_summary_stats_single(sub_ts)
        res.name = col
        results.append(res)
        
    # Combine the results from each column into a DataFrame and return the result
    output = pd.concat(results, axis=1).T
    if len(results) == 1:
        # If there is only one time series, use its input name to avoid
        #    strange column names being used by DataFrame conversion
        output.index = [results[0].name]
    return output

def _get_max_drawdown_summary_stats_single(mdd_ts):
    """ Calculate the summary statistics for the maximum draw-down of a single column.
        Namely, get the max-drawdown value, the start/trough/end dates, and the duration (in years).
        The function returns a pandas DataFrame
        """
    # Get the location of the maximum-draw doan. If there are multiple draw-downs with the same size,
    #     then take the most recent
    mdd_val = mdd_ts.min()
    t_mdd = np.where(mdd_ts == mdd_val)[0][-1]

    # Find the location of the high-water mark
    t_start = t_mdd - 1
    while not np.isclose(0, mdd_ts.iloc[t_start]):
        t_start -= 1

    # Find the time when the asset was no longer underwater following the draw-down
    t_end = t_mdd + 1
    while t_end < mdd_ts.shape[0] and not np.isclose(0, mdd_ts.iloc[t_end]):
        t_end += 1

    # Find the index of the start/trough/end of the draw-down
    idx_start = mdd_ts.index[t_start]
    idx_mdd = mdd_ts.index[t_mdd]

    if t_end < mdd_ts.shape[0]:
        idx_end = mdd_ts.index[t_end]
    else:
        idx_end = None

    # Find the duration of the draw-down
    if idx_end is not None:
        duration = secdb.tools.freq.get_years_between_dates(idx_start, idx_end)
    else:
        duration = secdb.tools.freq.get_years_between_dates(idx_start, mdd_ts.index[-1])

    # Summarize the results
    output_ts = pd.Series(dict(value=mdd_val,
                               start=idx_start,
                               trough=idx_mdd,
                               end=idx_end, 
                               duration=duration))
    output_ts.name = mdd_ts.name
    return output_ts

def get_num_years(ts):
    T = ts.index.values[-1] - ts.index.values[0]
    return np.timedelta64(T, 's') / np.timedelta64(1, 's') / secdb.tools.freq.SECONDS_PER_YEAR

def _vol_generic(filter_type, ts, stype, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, use_log_rtns=True, skipna=True):
    if use_log_rtns:
        rtns = convert_ts_type(ts, orig_type=stype, target_type=TS_TYPE_LOG_RETURNS, sampling_freq=sampling_freq)
    else:
        rtns = convert_ts_type(ts, orig_type=stype, target_type=TS_TYPE_SIMPLE_RETURNS, sampling_freq=sampling_freq)

    periods_per_year = secdb.tools.freq.get_periods_per_year(sampling_freq)
    rtn_vals = rtns.values    
    if isinstance(rtns, pd.Series):
        rtn_vals = rtn_vals[:,np.newaxis]
    
    risk_vals = []
    for j in range(rtn_vals.shape[1]):
        ts_j = rtn_vals[:,j]
        if skipna:
            ts_j = ts_j[~np.isnan(ts_j)]  # Remove NaN's
        if filter_type == 'none':
            filtered_vals = ts_j
        elif filter_type == 'negative':
            filtered_vals = ts_j[ts_j < 0]
        elif filter_type == 'positive':
            filtered_vals = ts_j[ts_j > 0]
        else:
            raise ValueError(f'Unsupported filter type: {filter_type}')
            
        if filtered_vals.size:
            risk_vals.append(filtered_vals.std() * np.sqrt(periods_per_year))
        else:
            risk_vals.append(0)

    if isinstance(ts, pd.Series):
        return pd.Series(risk_vals, index=[ts.name])
    else:
        return pd.Series(risk_vals, index=ts.columns)

def volatility(ts, stype, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, use_log_rtns=True, skipna=True):
    df = _vol_generic('none', ts=ts, stype=stype, sampling_freq=sampling_freq, use_log_rtns=use_log_rtns, skipna=skipna)
    df.name = 'volatility'
    return df

def downside_risk(ts, stype, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, use_log_rtns=True, skipna=True):
    df = _vol_generic('negative', ts=ts, stype=stype, sampling_freq=sampling_freq, use_log_rtns=use_log_rtns, skipna=skipna)
    df.name = 'downside_risk'
    return df

def upside_risk(ts, stype, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, use_log_rtns=True, skipna=True):
    df = _vol_generic('positive', ts=ts, stype=stype, sampling_freq=sampling_freq, use_log_rtns=use_log_rtns, skipna=skipna)
    df.name = 'upside_risk'
    return df

def drawdown(ts, stype):
    levels_ts = convert_ts_type(ts, orig_type=stype, target_type=TS_TYPE_LEVELS)
    drawdown_ts = -1 + levels_ts / levels_ts.expanding().max()
    drawdown_ts.ts_type = TS_TYPE_LEVELS
    return drawdown_ts

def max_drawdown(ts, stype):
    df = drawdown(ts, stype=stype).min(axis=0)
    if isinstance(ts, pd.Series):
        df = pd.Series(df, index=[ts.name])
    df.name = 'max_drawdown'
    return df

def VaR(ts, q, stype, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, horizon='12M'):
    # Convert the input time series to levels    
    levels = convert_ts_type(ts, orig_type=stype, target_type=TS_TYPE_LEVELS)
    
    # Calculate the number of periods in the measurement horizon, and then get rolling returns
    window = secdb.tools.freq.extract_window_size(horizon, levels)    
    rolling_returns = -1 + levels / levels.shift(window)
    
    df = rolling_returns.quantile(q)
    if isinstance(ts, pd.Series):
        df = pd.Series(df, index=[ts.name])    
    df.name = 'VaR_{}%'.format(q * 100)
    return df
    
def CVaR(ts, q, stype, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, horizon='12M'):
    # Convert the input time series to levels
    levels = convert_ts_type(ts, orig_type=stype, target_type=TS_TYPE_LEVELS)

    # Calculate the Value-at-Risk, which is required for the CVaR calculation
    _var = VaR(levels, q=q, stype=TS_TYPE_LEVELS, sampling_freq=sampling_freq, horizon=horizon)

    # Calculate the number of periods in the measurement horizon, and then get rolling returns
    window = secdb.tools.freq.extract_window_size(horizon, levels)
    rolling_returns = -1 + levels / levels.shift(window)
    
    # Get a matrix of the return values
    rtn_mtx = rolling_returns.values
    if isinstance(rolling_returns, pd.Series):
        rtn_mtx = rtn_mtx[:,np.newaxis]

    _cvar = []
    for j in range(rtn_mtx.shape[1]):
        ts_j = rtn_mtx[:,j]
        ts_j = ts_j[~np.isnan(ts_j)]
        if not len(ts_j):
            _cvar.append(np.nan)
        else:            
            _cvar_j = ts_j[ts_j <= _var[j]].mean()
            _cvar.append(_cvar_j)

    df = pd.Series(_cvar, index=_var.index)
    df.name = 'CVaR_{}%'.format(q * 100)
    return df    

def cum_return(ts, stype, skipna=True):
    n_years = get_num_years(ts)    
    levels_ts = convert_ts_type(ts, orig_type=stype, target_type=TS_TYPE_LEVELS)

    mtx = []
    mtx_vals = levels_ts.values
    if isinstance(levels_ts, pd.Series):
        mtx_vals = mtx_vals[:,np.newaxis]
        
    for j in range(mtx_vals.shape[1]):
        ts_j = mtx_vals[:,j]
        if skipna:
            ts_j = ts_j[~np.isnan(ts_j)]
        mtx.append(-1 + ts_j[-1] / ts_j[0])
        
    if isinstance(ts, pd.Series):
        return pd.Series(mtx, index=[ts.name], name='cum_return')
    else:
        return pd.Series(mtx, index=ts.columns, name='cum_return')
    
def ann_return(ts, stype, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, mean_type=GEOMETRIC_MEAN, skipna=True):
    samp_ts = convert_ts_type(ts, orig_type=stype, target_type=TS_TYPE_LEVELS, sampling_freq=sampling_freq)
    periods_per_year = secdb.tools.freq.get_periods_per_year(sampling_freq)
    if isinstance(samp_ts, pd.Series):
        cols = [samp_ts.name]
        samp_ts = pd.DataFrame(samp_ts)
        samp_ts.columns = cols
    
    if mean_type == ARITHMETIC_MEAN:
        if skipna:
            mtx = np.nan * np.ones((samp_ts.shape[1],), dtype=float)
            for j, col in enumerate(samp_ts.columns):
                rtns = -1 + samp_ts[col] / samp_ts[col].shift(periods=1)
                rtns = rtns.dropna()
                mtx[j] = rtns.mean() * periods_per_year
        else:
            rtns = -1 + samp_ts / samp_ts.shift(periods=1)
            mtx = rtns.loc[1:,:].mean(axis=0) * periods_per_year                
    elif mean_type == GEOMETRIC_MEAN:
        if skipna:
            mtx = np.nan * np.ones((samp_ts.shape[1],), dtype=float)
            for j, col in enumerate(samp_ts.columns):
                levels_ts = samp_ts[col].dropna()
                cum_perf = levels_ts.values[-1] / levels_ts.values[0]
                n_years = get_num_years(levels_ts)
                mtx[j] = -1 + np.power(cum_perf, 1 / n_years)
        else:        
            cum_perf = samp_ts.values[-1,:] / samp_ts.values[0,:]
            n_years = get_num_years(samp_ts)            
            mtx = -1 + np.power(cum_perf, 1 / n_years)
        
    name = f'ann_return_{mean_type}'
    if isinstance(ts, pd.Series):
        df = pd.Series(mtx.ravel(), index=[ts.name], name=name)
    else:
        df = pd.Series(mtx.ravel(), index=ts.columns, name=name)
    return df
                     
def calc_excess_returns(strategy_tr, benchmark_tr, stype, sampling_freq=None):
    benchmark_tr = _format_benchmark(benchmark_tr)    
    index = pd.DatetimeIndex.intersection(strategy_tr.index, benchmark_tr.index)
    strategy_rtns = convert_ts_type(strategy_tr.loc[index], orig_type=stype, 
                                        target_type=TS_TYPE_SIMPLE_RETURNS, sampling_freq=sampling_freq)
    benchmark_rtns = convert_ts_type(benchmark_tr.loc[index], orig_type=stype, 
                                        target_type=TS_TYPE_SIMPLE_RETURNS, sampling_freq=sampling_freq)
    if isinstance(strategy_rtns, pd.Series):
        ts = strategy_rtns - benchmark_rtns.values
    else:
        ts = strategy_rtns - benchmark_rtns.values[:,np.newaxis]
    return ts
                     
def ann_excess_return(tr, benchmark_tr, stype, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, mean_type=ARITHMETIC_MEAN):
    benchmark_tr = _format_benchmark(benchmark_tr)    
    excess_rtn = calc_excess_returns(tr, benchmark_tr, stype=stype, sampling_freq=sampling_freq)
    df = ann_return(excess_rtn, stype=TS_TYPE_SIMPLE_RETURNS, sampling_freq=sampling_freq, mean_type=mean_type)
    df.name = f'ann_exc_rtn_{mean_type}'
    return df
                     
def tracking_error(tr, benchmark_tr, stype, sampling_freq=DEFAULT_SAMPLING_FREQUENCY, skipna=True):
    benchmark_tr = _format_benchmark(benchmark_tr)
    excess_rtn = calc_excess_returns(tr, benchmark_tr, stype=stype, sampling_freq=sampling_freq)
    df = volatility(excess_rtn, stype=TS_TYPE_SIMPLE_RETURNS, sampling_freq=sampling_freq, skipna=skipna)
    df.name = 'tracking_error'
    return df

def information_ratio(tr, benchmark_tr, stype, sampling_freq=DEFAULT_SAMPLING_FREQUENCY):
    benchmark_tr = _format_benchmark(benchmark_tr)    
    er = ann_excess_return(tr, benchmark_tr=benchmark_tr, stype=stype, sampling_freq=sampling_freq, mean_type=ARITHMETIC_MEAN)
    te = tracking_error(tr, benchmark_tr=benchmark_tr, stype=stype, sampling_freq=sampling_freq)
    df = er / te
    df.name = 'info_ratio'
    return df    

def sharpe_ratio(tr, cash_tr, stype, sampling_freq=DEFAULT_SAMPLING_FREQUENCY):
    cash_tr = _format_benchmark(cash_tr)    
    df = information_ratio(tr, benchmark_tr=cash_tr, stype=stype, sampling_freq=DEFAULT_SAMPLING_FREQUENCY)
    df.name = 'sharpe_ratio'
    return df

def convert_ts_type(ts, orig_type, target_type, sampling_freq=None):
    inferred_freq = secdb.tools.freq.infer_freq(ts.index, allow_missing=True)
    if orig_type == target_type and (sampling_freq is None or sampling_freq == inferred_freq):
        new_ts = ts
    elif orig_type == TS_TYPE_LEVELS:
        if sampling_freq is not None and sampling_freq != '':
            sampled_ts = ts.resample(sampling_freq).ffill()
        else:
            sampled_ts = ts

        if target_type == TS_TYPE_LEVELS:
            new_ts = sampled_ts
        elif target_type in (TS_TYPE_SIMPLE_RETURNS, TS_TYPE_LOG_RETURNS):
            if isinstance(sampled_ts, pd.Series):
                sub_ts = sampled_ts.dropna()
                if target_type == TS_TYPE_SIMPLE_RETURNS:
                    new_ts = -1 + sub_ts / sub_ts.shift(1)
                else:
                    new_ts = np.log(sub_ts / sub_ts.shift(1))
            else:
                ts_list = []
                for col in sampled_ts.columns:
                    sub_ts = sampled_ts[col].dropna()
                    if target_type == TS_TYPE_SIMPLE_RETURNS:
                        ts_list.append(-1 + sub_ts / sub_ts.shift(1))
                    else:
                        ts_list.append(np.log(sub_ts / sub_ts.shift(1)))

                new_ts = pd.concat(ts_list, axis=1)
        else:
            raise ValueError(f'Conversion from {orig_type} to {target_type} is not supported')
    elif orig_type == TS_TYPE_SIMPLE_RETURNS:
        levels_ts = (1 + ts).cumprod(axis=0)
        _update_first_row_with_ones(levels_ts, inplace=True)
        new_ts = convert_ts_type(levels_ts, orig_type=TS_TYPE_LEVELS, 
                                    target_type=target_type, sampling_freq=sampling_freq)
    elif orig_type == TS_TYPE_LOG_RETURNS:
        levels_ts = np.exp(ts).cumprod(axis=0)
        _update_first_row_with_ones(levels_ts, inplace=True)
        new_ts = convert_ts_type(levels_ts, orig_type=TS_TYPE_LEVELS, 
                                    target_type=target_type, sampling_freq=sampling_freq)
    else:
        raise ValueError(f'Conversion from {orig_type} to {target_type} is not supported')
        
    # Update the ts_type information
    new_ts.ts_type = target_type
    return new_ts
        
def to_levels(ts, stype, sampling_freq=None):
    return convert_ts_type(ts, orig_type=stype, target_type=TS_TYPE_LEVELS, sampling_freq=sampling_freq)

def to_simple_returns(ts, stype, sampling_freq=None):
    return convert_ts_type(ts, orig_type=stype, target_type=TS_TYPE_SIMPLE_RETURNS, sampling_freq=sampling_freq)

def to_log_returns(ts, stype, sampling_freq=None):
    return convert_ts_type(ts, orig_type=stype, target_type=TS_TYPE_LOG_RETURNS, sampling_freq=sampling_freq)

def get_num_observations(ts):
    """ Calculate the number of non-NaN observations. """
    if isinstance(ts, pd.Series):
        cols = [ts.name]
        ts = pd.DataFrame(ts)
        ts.columns = cols
    
    count = np.zeros((ts.shape[1],), dtype=int)
    for j, col in enumerate(ts.columns):
        sub_ts = ts[col].dropna()
        count[j] = sub_ts.size
    return pd.Series(count, index=ts.columns, name='num_obs')

def get_first_valid_rows(ts):
    if isinstance(ts, pd.Series):
        if not isinstance(ts.index, pd.DatetimeIndex):
            raise ValueError("Only supported for objects inheriting from pandas DataFrame.")
        else:
            idx = ts.reset_index().dropna().index.values
            if idx.size:
                return idx[0]
            else:
                return None
    else:
        first_rows = [None] * len(ts.columns)
        for j, col in enumerate(ts.columns):
            idx = ts[col].first_valid_index()
            if idx is not None:
                first_rows[j] = np.where(idx == ts.index)[0][0]
        return pd.Series(first_rows, index=ts.columns, name='first_valid_rows')

def get_last_valid_rows(ts):
    if isinstance(ts, pd.Series):
        if not isinstance(ts.index, pd.DatetimeIndex):
            raise ValueError("Only supported for objects inheriting from pandas DataFrame.")
        else:
            idx = ts.reset_index().dropna().index.values
            if idx.size:
                return idx[-1]
            else:
                return None
    else:
        last_rows = [None] * len(ts.columns)
        for j, col in enumerate(ts.columns):
            idx = ts[col].last_valid_index()
            if idx is not None:
                last_rows[j] = np.where(idx == ts.index)[0][0]
        return pd.Series(last_rows, index=ts.columns, name='last_valid_rows')

def get_first_valid_index(ts):
    idx = get_first_valid_rows(ts)
    if isinstance(ts, pd.Series):
        labels = [ts.name]
    else:
        labels = ts.columns
    return pd.Series(ts.index[idx], index=labels, name='first_valid_index')

def get_last_valid_index(ts):
    idx = get_last_valid_rows(ts)
    if isinstance(ts, pd.Series):
        labels = [ts.name]
    else:
        labels = ts.columns    
    return pd.Series(ts.index[idx], index=labels, name='last_valid_index')

def _drop_nans(orig_ts, bf_ts):
    orig_ts_type = orig_ts.ts_type
    bf_ts_type = bf_ts.ts_type

    orig_ts = orig_ts.dropna()
    orig_ts.ts_type = orig_ts_type

    bf_ts = bf_ts.dropna()
    bf_ts.ts_type = bf_ts_type    
    return orig_ts, bf_ts

def get_backfilled_ts(orig_ts, bf_ts, stype, bf_type, bf_date=None, bf_fun=None):
    # Drop NaNs from the input time series
    orig_ts, bf_ts = _drop_nans(orig_ts, bf_ts)

    if bf_type == TS_TYPE_LEVELS:
        orig_rtns = to_simple_returns(orig_ts, stype=stype)
        bf_rtns = to_simple_returns(bf_ts, stype=bf_type)
        backfilled_rtns = get_backfilled_ts(orig_rtns, bf_rtns, stype=TS_TYPE_SIMPLE_RETURNS,
                                            bf_type=TS_TYPE_SIMPLE_RETURNS, bf_date=bf_date)
        backfilled_levels = to_levels(backfilled_rtns, stype=TS_TYPE_SIMPLE_RETURNS)
        backfilled_ts = backfilled_levels * orig_ts.values[-1] / backfilled_levels.loc[orig_ts.index[-1]]
        return backfilled_ts
    
    if isinstance(bf_fun, str):
        bf_fun = eval(bf_fun)
    
    if orig_ts.ts_type != bf_ts.ts_type:
        raise NotImplementedError("Not implemented for cases where 'ts_type' of the input time series are different.")

    backfilled_ts, bts = orig_ts.align(bf_ts, axis=0, join='outer')

    if bf_type in ('rates', 'returns'):
        pass
    elif bf_type == 'formula':
        bts = bf_fun(bts)
    else:
        raise ValueError(f'Unsupported backfill type: {bf_type}')

    if not bf_date:
        _date = orig_ts.index[0]
    else:
        _date = pd.Timestamp(bf_date)

    # Perform the actual back-filling
    idx = backfilled_ts.index < _date
    backfilled_ts.iloc[idx] = bts.values[idx].ravel()
    backfilled_ts.ts_type = orig_ts.ts_type
    return backfilled_ts

def _update_first_row_with_ones(levels_ts, inplace=True):
    if not inplace:
        levels_ts = levels_ts.copy()
    first_rows = get_first_valid_rows(levels_ts)
    if isinstance(levels_ts, pd.Series):
        levels_ts[first_rows] = 1.0
    else:
        for col in range(levels_ts.shape[1]):
            if first_rows[col] is not None and first_rows[col] > 0:
                levels_ts.iloc[int(first_rows[col]) - 1, col] = 1.0
    if not inplace:
        return levels_ts

def check_valid_ts_type(ts_type, allow_unknown=False):
    # Check that ts_type is present
    if not (ts_type in VALID_TS_TYPES or (allow_unknown and ts_type == TS_TYPE_UNKNOWN)):
        raise AttributeError("Attribute 'ts_type' is '{}' but must be one of {}".format(ts_type, VALID_TS_TYPES))
    else:
        return True    

def convert_frequency(ts, ts_types, target_frequency):
    if isinstance(ts_types, list):
        ts_types = np.array(ts_types)
    elif not isinstance(ts_types, np.ndarray):
        raise ValueError('Input "ts_types" must be a numpy array or a list, not a {}'.format(ts_types.__class__))

    # Initialize a DataFrame to contain the output
    # We initialize the columns so that if 'ts' has a MultiIndex, its dimensions will match output_ts
    output_ts = pd.DataFrame([], columns=ts.columns[0:0])
    
    # For time series that are levels, rates or dates, we can just take the end-of-period value
    idx = np.array([t in (TS_TYPE_LEVELS, TS_TYPE_RATES, TS_TYPE_DATES)  for t in ts_types], dtype=bool)
    if np.any(idx):
        sub_ts = ts.loc[:,idx]
        if target_frequency is not None and target_frequency != '':
            sub_ts = sub_ts.resample(target_frequency).last()
        output_ts = output_ts.merge(sub_ts, left_index=True, right_index=True, how='outer')

    # For returns, we must first convert to levels, then take the end-of-period value, and then convert back to returns
    for _ts_type in [TS_TYPE_SIMPLE_RETURNS, TS_TYPE_LOG_RETURNS]:
        idx = (ts_types == _ts_type)
        if np.any(idx):
            sub_ts = ts.loc[:,idx]
            sub_ts.ts_type = _ts_type
            levels = to_levels(sub_ts, stype=_ts_type, sampling_freq=target_frequency)
            rtn_ts = convert_ts_type(levels, orig_type=TS_TYPE_LEVELS, target_type=_ts_type, sampling_freq=target_frequency)
            output_ts = output_ts.merge(rtn_ts, left_index=True, right_index=True, how='outer')
    
    # Check that we processed all of the columns - it not, then there is some unsupported 'ts_type' for one of the series
    missing_tickers = set(ts.columns) - set(output_ts.columns)
    if missing_tickers:
        raise ValueError('Unkonwn series_type for some tickers: {}'.format(missing_tickers))
    else:
        return output_ts

def get_fundamental_frequencies(ts, stype=None):
    """ Find the fundamental frequency of each column in a time series.
    
        By 'fundamental frequency', we mean the frequency that this time
        series column would have it weren't combined with other columns,
        and we removed all missing and repeated data. For returns, this
        would also mean removing dates with 0 return.
    """ 
    frequencies = []
    for j, col in enumerate(ts.columns):
        # Drop all NaN values from this column
        sub_ts = ts[col].dropna()
        if not sub_ts.size:
            frequencies.append('')
        else:
            if stype is None or stype == TS_TYPE_UNKNOWN:
                idx = sub_ts.index
            elif stype in (TS_TYPE_SIMPLE_RETURNS, TS_TYPE_LOG_RETURNS):
                # Get rid of returns that are close to 0
                good_rows = ~np.isclose(0, sub_ts.values, atol=1e-6)
                idx = sub_ts.index[good_rows]
            else:
                # Get rid of repeated prices, rates, etc.
                good_rows = np.hstack([True, sub_ts.values[1:] != sub_ts.values[:-1]])
                idx = sub_ts.index[good_rows]

            # Add the inferred frequency to the list
            f = secdb.tools.freq.infer_freq(idx, allow_missing=True)
            frequencies.append(f)

    return frequencies
    
def unsmooth_returns(ts, stype, method=UNSMOOTH_METHOD_GELTNER, **kwargs):
    """ Unsmooth the returns, to remove serial auto-correlation. """
    if method in (UNSMOOTH_METHOD_GELTNER, UNSMOOTH_METHOD_GELTNER_ROLLING):
        return unsmooth_returns_numpy(ts, stype=stype, method=method, **kwargs)
    else:
        raise ValueError(f'Unsupported unsmoothing method: {method}')

def unsmooth_returns_numpy(ts, stype, method, skipna=True, window=None, max_beta=0.75, smoothing=0.9):
    """ Use unsmoothing techniques that take numpy inputs and apply them to time series.
    
        Apply a range of unsmoothing techniques, whose exact calculations should
        be carried out in another function. These unsmoothing techniques should
        support an input of an array of returns, with one or multiple columns.
    """
    orig_type = stype
    input_is_series = isinstance(ts, pd.Series)
    if input_is_series:
        cols = [ts.name]
        ts = pd.DataFrame(ts)
        ts.columns = cols

    # Get the simple returns
    rtns = convert_ts_type(ts, orig_type=stype, target_type=TS_TYPE_SIMPLE_RETURNS)

    # Get a numpy array of unsmoothed returns
    if method == UNSMOOTH_METHOD_GELTNER:
        uns_rtn_mtx = secdb.tools.stats.geltner_unsmooth(rtns.to_numpy(), skipna=skipna)
    elif method == UNSMOOTH_METHOD_GELTNER_ROLLING:
        if window is None:
            raise ValueError('The "window" argument must be specified to use Geltner rolling unsmoothing.')
        else:
            uns_rtn_mtx = secdb.tools.stats.geltner_unsmooth_rolling(rtns.to_numpy(), window, skipna=skipna,
                                                                     max_beta=max_beta, smoothing=smoothing)
    else:
        raise ValueError('Unsupported Geltner unsmoothing method.')
    
    uns_rtn_ts = pd.DataFrame(uns_rtn_mtx, index=rtns.index, columns=rtns.columns)
    
    # Convert back to the original time series type
    output_ts = convert_ts_type(uns_rtn_ts, orig_type=TS_TYPE_SIMPLE_RETURNS, target_type=stype)

    if input_is_series:
        return output_ts.iloc[:,0]
    else:
        return output_ts

def get_total_return_index_from_prices(prices, income):
    """ Combine the price and income into a total return index.
    """
    # Drop dates where there is no data
    prices, income = prices.dropna().align(income.dropna(), axis=0)

    # Calculate the price returns
    price_rtns = -1 + prices / prices.shift(1).values
    price_rtns.iloc[0] = 0.0

    # Calculate the dividend returns
    div_rtns = income / prices.shift(1).values
    div_rtns.iloc[0] = 0.0

    # Calculate the total return index
    tr_index = np.cumprod(1 + price_rtns + div_rtns) * prices.iloc[0]
    tr_index.name = f'{base_ticker}(RI)'

    # Combine the old and new time series and return the result
    combined_ts = pd.concat([ts, tr_index], axis=1)
    return combined_ts    
    
def _format_benchmark(bmk):
    """ Make sure the benchmark is a pandas Series object (rather than a levels object)
    """
    # Make sure that the bmk time series is 1-d
    if bmk.ndim == 2:
        assert bmk.shape[1] == 1, 'Benchmark should be a pandas Series or a DataFrame with only a single column.'
        bmk = bmk.iloc[:,0]
    return bmk