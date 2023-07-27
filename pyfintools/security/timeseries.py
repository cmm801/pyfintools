""" Provides unified access to time series and meta data for multiple securities.

    The motivation behind this module is to extend the pandas DataFrame class in 
    order to include meta data for the time series, as well as to enable asset-class
    specific calculations like currency conversion, yield curve manipulations, and
    converting price series ('levels') to returns and vice versa.
    Pandas DataFrame is difficult to extend via simple inheritance, and so we create
    a brand new class which mimics many functions of DataFrame, but allows the inclusion
    of meta data.
    
    In other respects, the timeseries module is set up in a parallel fashion to the 
    'single' and 'panel' modules, also in the 'security' package. The TimeSeries class
    in this module is comparable to the Security class in 'single' and the Panel class 
    in 'panel'. The base class TimeSeries is inherited by all subclasses,
    which extend its variables and functionalities depending on the specifics
    of the new instrument.
    
    Classes that inherit from Asset and AssetIndex possess the ability to perform currency
    conversion and currency hedging. These currency manipulations are in turm performed
    by FX class objects, which also inherit from TimeSeries.

    All classes defined in this module can be initialized via the constructor by providing
    two arguments (both of which can be None):
        ts: a pandas DataFrame containing time series information. The columns of 'ts' should
            be the ticker codes. The index of the DataFrame should be the dates/times of the
            observations. The columns of 'ts' must match the columns of 'meta'.
        meta: a pandas DataFrame containing information about the time series data.
            The columns of 'meta' should match the columns of 'ts'.
            The index of the 'meta' DataFrame are the attribute names for a given type of meta data.
            For example, some standard index names would be 'sec_code', 'category_1_code', 'ticker_code'.
"""

import numpy as np
import pandas as pd
import scipy.interpolate
import talib
from collections import defaultdict, Iterable

import pyfintools.tools.utils
import pyfintools.tools.fts
from pyfintools.security.constants import TENOR_SPOT, DEFAULT_CROSS_CURRENCY, DEFAULT_FX_HEDGING_FREQUENCY


# Define some constants to categorize the types of currency conversion / hedging we may encounter
_COMPLETE = 0
_UNKNOWN = 1
_HDG_PARTIAL = 2
_HDG_TO_HDG = 3
_UNH_TO_HDG = 4
_UNH_TO_UNH = 5
_HDG_TO_UNH = 6

# Define rate_type values, for use in the YieldCurve
RATE_TYPE_ZERO = 'zero'
RATE_TYPE_PAR = 'par'
RATE_TYPE_FORWARD = 'forward'


class TimeSeries(object):
    """ Provides unified access to time series and meta data for multiple securities.

        The motivation behind this class is to extend the pandas DataFrame class in 
        order to include meta data for the time series, as well as to enable asset-class
        specific calculations like currency conversion, yield curve manipulations, and
        converting price series ('levels') to returns and vice versa.
        Pandas DataFrame is difficult to extend via simple inheritance, and so we create
        a brand new class which mimics many functions of DataFrame, but allows the inclusion
        of meta data.

        All classes defined in this module inherit from TimeSeries, and can be initialized 
        via the constructor by providing two arguments (both of which can be None):
            ts: a pandas DataFrame containing time series information. The columns of 'ts' should
                be the ticker codes. The index of the DataFrame should be the dates/times of the
                observations. The columns of 'ts' must match the columns of 'meta'.
            meta: a pandas DataFrame containing information about the time series data.
                The columns of 'meta' should match the columns of 'ts'.
                The index of the 'meta' DataFrame are the attribute names for a given type of meta data.
                For example, some standard index names would be 'sec_code', 'category_1_code', 'ticker_code'.
    """
    _SUPPORTED_TYPES_FOR_ARITHMETIC = (int, float, np.float32, np.ndarray, pd.Series, pd.DataFrame)
    
    def __init__(self, ts, meta):
        """ Initialize a TimeSeries object.
            
            Arguments:
                ts: a pandas DataFrame containing time series information. The columns of 'ts' should
                    be the ticker codes. The index of the DataFrame should be the dates/times of the
                    observations. The columns of 'ts' must match the columns of 'meta'.
                meta: a pandas DataFrame containing information about the time series data.
                    The columns of 'meta' should match the columns of 'ts'.
                    The index of the 'meta' DataFrame are the attribute names for a given type of meta data.
                    For example, some standard index names would be 'sec_code', 'category_1_code', 'ticker_code'.
        """        
        super(TimeSeries, self).__init__()
        self.ts = ts
        self.meta = meta
        
    def __getitem__(self, val):
        return self._constructor(self.ts[val], self.meta[val])

    def _constructor(self, _ts, _meta):
        return self.__class__(_ts, _meta)

    def __str__(self):
        return self.ts.__str__()

    def __repr__(self):
        return super().__repr__()    

    @property
    def ts(self):
        return self._ts
        
    @ts.setter
    def ts(self, _timeseries):
        """ Setter function for 'ts', ensures that 'ts' is a pandas DataFrame with a DatetimeIndex. """
        if isinstance(_timeseries, pd.Series):
            _timeseries = pd.DataFrame(_timeseries)
        elif not isinstance(_timeseries, pd.DataFrame):
            raise ValueError('Unsupported time series class type: {}'.format(_timeseries.__class__))
        
        if _timeseries.columns.duplicated().any():
            raise ValueError('Column names must be unique.')
        else:
            # Save a copy of the Dataframe
            self._ts = _timeseries.copy()

            # Ensure the index is a pandas DatetimeIndex
            self._ts.index = pd.DatetimeIndex(_timeseries.index)
        
    @property
    def meta(self):
        return self._meta
        
    @meta.setter
    def meta(self, _metadata):
        """ Setter function for 'meta' ensures that 'meta' is a pandas DataFrame. """
        if isinstance(_metadata, pd.Series):
            _metadata = pd.DataFrame(_metadata)
            
        if not isinstance(_metadata, pd.DataFrame):
            raise ValueError('Unsupported meta data class type: {}'.format(_metadata.__class__))
            
        if set(self.ts.columns) != set(_metadata.columns):
            raise ValueError('Column names must be the same for meta and time series data.')
        elif _metadata.columns.duplicated().any():
            raise ValueError('Column names must be unique.')
        else:
            self._meta = _metadata[self.ts.columns]
        
    @property
    def shape(self):
        return self.ts.shape
    
    @property
    def columns(self):
        return self.ts.columns
    
    @columns.setter
    def columns(self, cols):
        if isinstance(cols, str) or isinstance(cols, tuple) or not isinstance(cols, Iterable):
            cols = [cols]

        self._ts.columns = cols
        self._meta.columns = cols

    @property
    def index(self):
        return self.ts.index

    @index.setter
    def index(self, idx):
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError('The TimeSeries index must always be of type pandas DatetimeIndex.')
        else:
            self._ts.index = idx
    
    @property
    def values(self):
        return self.ts.values
    
    @values.setter
    def values(self, vals):
        self._ts.values = vals

    @property
    def loc(self):
        return IndexSliceHelper(self, 'loc')
    
    @property
    def iloc(self):
        return IndexSliceHelper(self, 'iloc')
    
    @property
    def sec_code(self):
        return self._get_property_value('sec_code')

    @property
    def series_type_code(self):
        return self._get_property_value('series_type_code')

    @property
    def series_type_name(self):
        return self._get_property_value('series_type_name')

    @property
    def name(self):
        return self._get_property_value('name')
    
    @property
    def category_1_code(self):
        return self._get_property_value('category_1_code')
    
    @property
    def category_2_code(self):
        return self._get_property_value('category_2_code')

    @property
    def category_3_code(self):
        return self._get_property_value('category_3_code')

    @property
    def is_tradable(self):
        return self._get_property_value('is_tradable', None)

    @property
    def ts_type(self):
        return self._get_property_value('ts_type', '')
    
    @ts_type.setter
    def ts_type(self, val):
        pyfintools.tools.fts.check_valid_ts_type(val, allow_unknown=True)
        self.meta.loc['ts_type'] = val
    
    @ts_type.deleter
    def ts_type(self):
        self.ts_type = pyfintools.tools.fts.TS_TYPE_UNKNOWN

    def _get_property_value(self, attr, default_val=''):
        if attr in self.meta.index:
            return self.meta.loc[attr]
        else:
            return pd.Series([default_val] * len(self.columns), 
                             index=self.meta.columns, 
                             dtype=default_val.__class__)

    @property
    def frequency(self):
        """ From the DatetimeIndex on the 'ts' variable, infer the frequency of data observations. 
        
            The output will be a single character, corresponding to the pandas classification of
            time series frequency (e.g. 'D', 'B', 'W', 'M', 'Y', etc.). 
            If no frequency can be inferred, then None will be returned.
        """
        return pyfintools.tools.freq.infer_freq(self.index)

    @property
    def periods_per_year(self):
        """ Calculate the number of observations per year, given the time series' frequency. """
        return pyfintools.tools.freq.get_periods_per_year(self.frequency)

    def copy(self):
        """ Make a copy of the time series. """
        return self._constructor(self.ts.copy(), self.meta.copy())

    def align(self, other, **kwargs):
        """ Align the indices of two different TimeSeries objects. 
        
            This function overloads the pandas 'align' function. You can see pandas for more documentation.
        """
        ts_self, ts_other = self.ts.align(other.ts, **kwargs)
        new_self = self._constructor(ts_self, self.meta[ts_self.columns].copy())
        new_other = other._constructor(ts_other, other.meta[ts_other.columns].copy())
        return new_self, new_other

    def head(self, window=5, **kwargs):
        """ Keep just the first time series rows, determined by window. 
            Argument "window" can be an integer or a string (e.g. 12, or "1y", "3M", etc.) """
        n = pyfintools.tools.freq.extract_window_size(window, self.ts)
        return self._constructor(self.ts.head(n, **kwargs), self.meta.copy())

    def tail(self, window=5, **kwargs):
        """ Keep just the last time series rows, determined by window. 
            Argument "window" can be an integer or a string (e.g. 12, or "1y", "3M", etc.) """        
        n = pyfintools.tools.freq.extract_window_size(window, self.ts)        
        return self._constructor(self.ts.tail(n, **kwargs), self.meta.copy())

    def ffill(self, **kwargs):
        """ Fill missing observations by using the previous available observations. """
        return self._constructor(self.ts.ffill(**kwargs), self.meta.copy())
    
    def append(self, df, **kwargs):
        """ Append additional time series data onto 'ts'.
        
            The index of the input DataFrame cannot overlap with the index of the original object.
            Arguments:
                df: (DataFrame) a pandas DataFrame, with a DatetimeIndex
        """
        if pd.DatetimeIndex.intersection(df.index, self._ts.index).size:
            raise ValueError('Cannot append data whose index overlaps with the current index.')
        else:
            new_ts = self._ts.append(df)
            new_ts = new_ts.sort_index()
            return self._constructor(new_ts, self.meta.copy())
    
    def insert_empty_rows(self, idx, **kwargs):
        """ Insert empty rows into the time series portion of the object. """
        if not isinstance(idx, Iterable):
            idx = [idx]

        empty_data = np.nan * np.ones((len(list(idx)), self._ts.shape[1]), dtype=float)
        df = pd.DataFrame(empty_data, index=idx, columns=self.columns)
        return self.append(df)        
    
    def merge(self, right, same_class=True):
        """ Merge the time series and meta DataFrame objects.
        
            Arguments
              right: the other TimeSeries object on which to join
              same_class: True/False - whether to permit the merger of two objects from different TimeSeries subclasses.
              
            Returns a new TimeSeries security object, with combined meta and time series data.
        """
        if not isinstance(right, TimeSeries):
            # If the input is a list, tuple or array, then loop through and merge all the inputs
            merged_ts = self
            for r in right:
                merged_ts = merged_ts.merge(r, same_class=same_class)
            return merged_ts
        else:
            # ...otherwise, if the input is a single TimeSeries object, perform the merge
            if self.__class__ != right.__class__:
                if same_class:
                    raise ValueError('Objects must be of the same class in order to merge.')
                else:
                    constructor_handle = TimeSeries
            else:
                constructor_handle = self._constructor

            if (set(self.ts.columns) & set(right.ts.columns)):
                raise ValueError('Merge cannot be called when the right and left objects share column names.')

            merged_ts = self.ts.merge(right.ts, left_index=True, right_index=True, how='outer')
            merged_meta = pd.concat([self.meta, right.meta], axis=1)
            return constructor_handle(merged_ts, merged_meta)

    def dropna(self, **kwargs):
        """ Drop missing data, following the convention of the pandas library. """
        adj_ts = self.ts.dropna(**kwargs)
        return self._constructor(adj_ts, self.meta[adj_ts.columns].copy())        

    def shift(self, *args, **kwargs):
        """ Shift the time series index, following the convention of the pandas library. """
        return self._constructor(self.ts.shift(*args, **kwargs), self.meta.copy())

    def mean(self, *args, **kwargs):
        """ Compute the mean of the time series, following the convention of the pandas library. """        
        return self.ts.mean(*args, **kwargs)

    def median(self, *args, **kwargs):
        """ Compute the median of the time series, following the convention of the pandas library. """                
        return self.ts.median(*args, **kwargs)

    def std(self, *args, **kwargs):
        """ Compute the standard deviation of the time series, following the convention of the pandas library. """
        return self.ts.std(*args, **kwargs)

    def var(self, *args, **kwargs):
        """ Compute the variance of the time series, following the convention of the pandas library. """
        return self.ts.var(*args, **kwargs)

    def summary_stats(self, skipna=True, bmk=None):
        """ Compute summary statistics for the TimeSeries object.
        
            The details of which summary statistics are computed are contained in the 'fts' module.
        """
        if isinstance(bmk, TimeSeries):
            bmk_ts = bmk.ts
        else:
            bmk_ts = bmk
        sampling_freq = self.frequency
        return self._apply_fts_methods('summary_stats', sampling_freq=sampling_freq, skipna=skipna, bmk=bmk_ts)

    def cum_return(self, skipna=True):
        """ Compute the cumulative return of the time series. """
        return self._apply_fts_methods('cum_return')

    def ann_return(self, mean_type=pyfintools.tools.fts.GEOMETRIC_MEAN, skipna=True):
        """ Compute the annual return of the time series. """
        sampling_freq = self.frequency        
        return self._apply_fts_methods('ann_return', sampling_freq=sampling_freq, mean_type=mean_type, skipna=skipna)

    def volatility(self, use_log_rtns=True, skipna=True):
        """ Compute the annualized volatility of the time series. """
        sampling_freq = self.frequency
        return self._apply_fts_methods('volatility', sampling_freq=sampling_freq, use_log_rtns=use_log_rtns, skipna=skipna)

    def downside_risk(self, use_log_rtns=True, skipna=True):
        """ Compute the downside standard deviation. """
        sampling_freq = self.frequency
        return self._apply_fts_methods('downside_risk', sampling_freq=sampling_freq, use_log_rtns=use_log_rtns, skipna=skipna)

    def upside_risk(self, use_log_rtns=True, skipna=True):
        """ Compute the upside standard deviation.  """
        sampling_freq = self.frequency
        return self._apply_fts_methods('upside_risk', sampling_freq=sampling_freq, use_log_rtns=use_log_rtns, skipna=skipna)

    def drawdown(self):
        """ Compute the draw-down time series. """
        return self._apply_fts_methods('drawdown')

    def max_drawdown(self):
        """ Compute the maximum draw-down. """
        return self._apply_fts_methods('max_drawdown')

    def VaR(self, q, horizon='12M'):
        """ Value-at-Risk (VaR). 
            Arguments:
                q: (float or list/numpy array  of floats) indicates the quantile for which we calculate the value at risk. 
        """
        return self._apply_fts_methods('VaR', q=q, horizon=horizon)

    def CVaR(self, q, horizon='12M'):
        """ Conditional Value-at-Risk (CVaR). Also known as Expected Shortfall .
            Arguments:
                q: (float or list/numpy array  of floats) indicates the quantile for which we calculate the value at risk. 
        """
        return self._apply_fts_methods('CVaR', q=q, horizon=horizon)

    def unsmooth_returns(self, method=pyfintools.tools.fts.UNSMOOTH_METHOD_GELTNER, **kwargs):
        """ Unsmooth the returns/levels to remove auto-correlation. """
        return self._apply_fts_methods('unsmooth_returns', method=method, **kwargs)

    @property
    def num_observations(self):
        """ Find the number of observations in the time series (e.g. the number of rows) """
        return self._apply_fts_methods('num_observations')
    
    @property
    def first_valid_rows(self):
        """ Returns a list of the first non-NaN row for each column. """
        return self._apply_fts_methods('first_valid_rows')

    @property
    def first_valid_index(self):
        """ Returns a list of the first non-NaN index for each column. """
        return self._apply_fts_methods('first_valid_index')        

    @property    
    def last_valid_rows(self):
        """ Returns a list of the last non-NaN row for each column. """        
        return self._apply_fts_methods('last_valid_rows')

    @property
    def last_valid_index(self):
        """ Returns a list of the last non-NaN index for each column. """        
        return self._apply_fts_methods('last_valid_index')

    @property
    def fundamental_frequency(self):
        """ Returns a list of the fundamental frequencies of each column.
        
            The fundamental frequency is the frequency of the time series data 
            in the individual column if all NaNs were removed, as well as all
            repeated values (for 'levels' and 'rates') or all 0's (for 'returns').
        """
        return self._apply_fts_methods('fundamental_frequency')

    def _apply_fts_methods(self, fun_name, *args, **kwargs):
        """ Use methods from the 'fts' module and apply them to the time series data. """
        
        # First, split the existing TimeSeries object into sub-objects, if the instance variable "ts_type" is not unique
        _ts_types = self.ts_type
        uniq_ts_types = set(_ts_types.values)
        pd_result_type = None  # Keep track of the type of pandas object in the results
        results = []
        for uniq_ts_type in uniq_ts_types:
            sub_ts = self.ts[_ts_types.where(_ts_types == uniq_ts_type).index]
            sub_ts.ts_type = uniq_ts_type
            fun_handle = getattr(sub_ts.fts, fun_name) 
            res = fun_handle(*args, **kwargs)
            if pd_result_type is None:
                pd_result_type = res.__class__
            elif not isinstance(res, results[-1].__class__):
                raise ValueError('Inconsistent pandas types in result: {} and {}'.format(res.__class__, results[-1].__class__))
            results.append(res)
        
        # Format the output
        if isinstance(res, list):
            return results[0]
        elif not hasattr(res, 'index'):
            raise ValueError('Unsupported output type. Was expecting an object with an "index" variable.')
        elif not isinstance(res.index, pd.DatetimeIndex):
            # If the outputs are pandas Series objects, combine them and return a pandas Series
            res_ts = pd.concat(results, axis=0)
            return res_ts.loc[self.columns]
        elif isinstance(res, pd.DataFrame):
            # If the objects are pandas DataFrame objects, combine them
            res_ts = pd.concat(results, axis=1)
            res_ts = res_ts[self.columns]

            # Get the new ts_types, as these may have changed depending on the calculation
            new_ts_type = None
            for r in results:
                if not hasattr(r, 'ts_type'):
                    curr_ts_type = pyfintools.tools.fts.TS_TYPE_UNKNOWN
                else:
                    curr_ts_type = r.ts_type

                if new_ts_type is None:
                    new_ts_type = curr_ts_type
                elif new_ts_type != curr_ts_type:
                    raise ValueError('All calculation results should yield a DataFrame with the same ts_type.')
            
            # Create a new time series object with the results
            ts_obj = self._constructor(res_ts, self.meta.copy())
            ts_obj.ts_type = new_ts_type
            return ts_obj
        else:
            raise ValueError('Unknown result type: {}'.format(pd_result_type.__class__))
    
    def plot(self, *args, **kwargs):
        """ Plot the time series data. """
        return self.ts.plot(*args, **kwargs)

    def log(self):
        """ Take the log of the time series. """
        return self._constructor(np.log(self.ts), self.meta.copy())

    def cov(self, **kwargs):
        """ Calculate the covariance of the time series. """
        return self.ts.cov(**kwargs)

    def cov(self, **kwargs):
        """ Calculate the correlation of the time series. """
        return self.ts.corr(**kwargs)
    
    def to_ts_type(self, target_type=pyfintools.tools.fts.TS_TYPE_LEVELS, sampling_freq=None):
        """ Convert the time series data to a new 'ts_type' 
        """
        # Get the original column order so we can put things back into the appropriate order
        orig_cols = self.columns
        
        # Go through the different ts_types and adjust them accordingly
        uniq_ts_types = set(self.ts_type)
        self_copy = self.copy()
        new_ts_TimeSeries = []
        for ts_type in uniq_ts_types:
            sub_meta = self_copy.meta.iloc[:,self_copy.ts_type.values == ts_type]
            sub_ts = self_copy.ts[sub_meta.columns]
            sub_ts.ts_type = ts_type
            new_sub_ts = sub_ts.fts.convert_ts_type(target_type, sampling_freq=sampling_freq)
            new_ts_TimeSeries.append(new_sub_ts)

        # Concatenate the adjusted time series, and update the meta data
        self_copy.ts = pd.concat(new_ts_TimeSeries, axis=1)
        self_copy.meta.loc['ts_type'] = target_type
        
        # Create a new object with the converted data
        return self_copy[orig_cols]
    
    def convert_frequency(self, target_freq):
        """ Convert the frequency of the TimeSeries to a new target frequency. """
        orig_ts = self.ts
        current_freq = list(set(self.ts_type))
        if len(current_freq) > 1:
            raise NotImplementedError('Not implemented when columns have more than 1 ts_type.')
        else:
            orig_ts.ts_type = current_freq[0]
            orig_ts.fts.convert_frequency(target_freq) 
            new_ts = self.ts.fts.convert_frequency(target_freq)
        return self._constructor(new_ts, self.meta.copy())
    
    def to_levels(self, sampling_freq=None):
        """ Convert the time series data to 'levels' (e.g. from returns to price levels) """
        return self.to_ts_type(target_type=pyfintools.tools.fts.TS_TYPE_LEVELS, sampling_freq=sampling_freq)

    def to_simple_returns(self, sampling_freq=None):
        """ Convert the time series to simple (arithmetic) return. """
        return self.to_ts_type(target_type=pyfintools.tools.fts.TS_TYPE_SIMPLE_RETURNS, sampling_freq=sampling_freq)

    def to_log_returns(self, sampling_freq=None):
        """ Convert the time series to log return. """
        return self.to_ts_type(target_type=pyfintools.tools.fts.TS_TYPE_LOG_RETURNS, sampling_freq=sampling_freq)

    def sort_columns(self, attrib_name, reverse=False):
        """ Sort the columns in alphabetical order. """
        attrib_vals = self.meta.loc[attrib_name]
        idx = list(np.argsort(attrib_vals))
        if reverse:
            idx = idx[::-1]

        # Return a copy of the object
        cols = self.columns[idx]
        return self.copy()[cols]

    def query(self, qry_str):
        """Apply a SQL-like query to the meta data. 
        
           This method is modeled on the pandas DataFrame 'query' method. The pandas documentaion
           will contain more details on usage.
        """
        md = self._meta.T.query(qry_str).T
        return self._constructor(self._ts[md.columns], self._meta[md.columns])
    
    def apply_filter(self, attrib_name, target_values):
        """ Apply a filter to keep only time series columns that satisfy the constraints.
        
            Arguments:
                attrib_name: (str) the meta-data field on which we will perform the filter
                target_values: (any) this function will keep only columns whose meta data
                    matches one of the target values.
                    
            Output: A new TimeSeries object with a subset of the original columns (possibly empty)
        """
        if isinstance(target_values, (list, set)):
            cols = self.columns[[x in target_values for x in self.meta.loc[attrib_name]]]
        else:
            cols = self.columns[[x == target_values for x in self.meta.loc[attrib_name]]]

        # Return a copy of the object
        return self.copy()[cols]

    def drop(self, attrib_name, target_values):
        """ Apply a filter to drop time series columns that match the target values.

            Arguments:
                attrib_name: (str) the meta-data field on which we will perform the filter
                target_values: (any) this function will drop all columns whose meta data
                    matches one of the target values.
                    
            Output: A new TimeSeries object with a subset of the original columns (possibly empty)
        """        
        if isinstance(target_values, (list, set)):
            cols = self.columns[[x not in target_values for x in self.meta.loc[attrib_name]]]
        else:
            cols = self.columns[[x != target_values for x in self.meta.loc[attrib_name]]]
            
        # Return a copy of the object
        return self.copy()[cols]
        
    def select(self, attrib_name, target_values):
        """ Apply a filter to select a specific subset of values.
        
            There should be only one column whose meta data matches each of the target values.
            For example, assume that one of the attribute names is 'risk_currency', and 
            the time series object has currency = ['USD', 'EUR', 'JPY', 'CHF', 'EUR', 'USD']
            Then, calling 
            >> obj.select('risk_currency', 'JPY')
            will return a new TimeSeries object corresponding to the 3rd column, but
            >> obj.select('risk_currency', 'USD')
            will raise an exception, because the first and last column both have 'USD' as their 
            risk currency, but this function requires a unique match.
            Finally, if we tried
            >> obj.select('risk_currency', ['JPY', 'CHF'])
            then a new TimeSeries object containing the 3rd and 4th columns of the original object
            would be returned.
            
            Arguments:
                attrib_name: (str) the meta-data field on which we will perform the filter
                target_values: (any) this function will keep all columns whose meta data
                    matches one of the target values, as long as there are not multiple matches.
        """ 
        attrib_vals = self.meta.loc[attrib_name]
        if not isinstance(target_values, Iterable) or isinstance(target_values, str):
            target_values = [target_values]

        locations = dict()
        for j, val in enumerate(attrib_vals):
            if val in target_values:
                if val not in locations:
                    locations[val] = j
                else:
                    raise ValueError(f'Multiple securities with the same attribute value ({val}) are not permitted.')

        missing = set(target_values) - set(locations.keys())                    
        if missing:
            raise ValueError('Missing some target values: {}'.format(list(missing)))
        elif len(target_values) != len(locations):
            raise ValueError('Columns must be unique.')
        else:
            idx = [locations[val] for val in target_values]
            cols = self.columns[idx]

        # Return a copy of the object
        return self.copy()[cols]

    def cast(self, class_handle):
        """ Cast from one TimeSeries subclass to a different TimeSeries subclass. 
        
            Typical use case:
            If we start with an object 'obj' that is a TimeSeries object, then we can
            cast to an 'Asset' object by calling
            >> obj.cast('Asset')
            
            Arguments:
                class_handle: (str) the string name of the TimeSeries subclass to which
                    we want to cast the original object.
        """
        if isinstance(class_handle, str):
            class_handle = eval(class_handle)
            
        if class_handle == MonthlyYieldCurve:
            return self._cast_to_MonthlyYieldCurve()
        else:
            return class_handle(self.ts.copy(), self.meta.copy())

    def _cast_to_MonthlyYieldCurve(self):
        if not isinstance(self, (InterestRate, YieldCurve)):
            raise ValueError('Can only cast InterestRate or YieldCurve objects to MonthlyYieldCurve.')
        else:
            if isinstance(self, InterestRate):
                yc = self.cast(YieldCurve)
            else:
                yc = self

            # Get all of the monthly rates between the min and max tenor, inclusive
            tenor_min = np.floor(12 * np.min(self.tenor_in_years))
            tenor_max = np.ceil(12 * np.max(self.tenor_in_years))
            target_tenors_in_months = np.arange(tenor_min, tenor_max + 1)
            meta_list = []
            for tnr in target_tenors_in_months:
                tmp = dict(tenor=f'{tnr}m', tenor_in_months=tnr, tenor_in_years=tnr/12,
                           risk_currency=yc.risk_currency[0], denominated_currency=yc.denominated_currency[0],
                           compounding_freq=yc.compounding_freq, rate_type=yc.curve_type)
                meta_list.append(tmp)

            # Create the meta DataFrame
            _meta = pd.DataFrame.from_dict(meta_list).T
            _meta.columns = _meta.loc['tenor_in_months'].values

            # Create the time series DataFrame
            _ts = yc.get_yields(target_tenors_in_months/12)
            _ts.columns = target_tenors_in_months
            return MonthlyYieldCurve(_ts, _meta)
    
    # Overload operators
    def __neg__(self):
        obj = self._constructor(-self.ts.copy(), self.meta.copy())
        del obj.ts_type
        return obj
        
    def __add__(self, right):
        if not isinstance(right, self._SUPPORTED_TYPES_FOR_ARITHMETIC):
            raise ValueError('Unsupported type: {}'.format(right.__class__))
        else:
            obj = self._constructor(self.ts + right, self.meta.copy())
            del obj.ts_type
            return obj            

    def __radd__(self, right):
        if not isinstance(right, self._SUPPORTED_TYPES_FOR_ARITHMETIC):
            raise ValueError('Unsupported type: {}'.format(right.__class__))
        else:
            obj = self._constructor(self.ts + right, self.meta.copy())
            del obj.ts_type
            return obj            
        
    def __sub__(self, right):
        if not isinstance(right, self._SUPPORTED_TYPES_FOR_ARITHMETIC):
            raise ValueError('Unsupported type: {}'.format(right.__class__))
        else:
            obj = self._constructor(self.ts - right, self.meta.copy())
            del obj.ts_type
            return obj            
        
    def __rsub__(self, right):
        if not isinstance(right, self._SUPPORTED_TYPES_FOR_ARITHMETIC):
            raise ValueError('Unsupported type: {}'.format(right.__class__))
        else:
            obj = self._constructor(right - self.ts, self.meta.copy())
            del obj.ts_type
            return obj

    def __mul__(self, right):
        if not isinstance(right, self._SUPPORTED_TYPES_FOR_ARITHMETIC):
            raise ValueError('Unsupported type: {}'.format(right.__class__))
        else:
            obj = self._constructor(self.ts * right, self.meta.copy())
            del obj.ts_type
            return obj            

    def __rmul__(self, right):
        if not isinstance(right, self._SUPPORTED_TYPES_FOR_ARITHMETIC):
            raise ValueError('Unsupported type: {}'.format(right.__class__))
        else:
            obj = self._constructor(self.ts * right, self.meta.copy())
            del obj.ts_type
            return obj            
        
    def __truediv__(self, right):
        if not isinstance(right, self._SUPPORTED_TYPES_FOR_ARITHMETIC):
            raise ValueError('Unsupported type: {}'.format(right.__class__))
        else:
            obj = self._constructor(self.ts / right, self.meta.copy())
            del obj.ts_type
            return obj            

    def __rtruediv__(self, right):
        if not isinstance(right, self._SUPPORTED_TYPES_FOR_ARITHMETIC):
            raise ValueError('Unsupported type: {}'.format(right.__class__))
        else:
            obj = self._constructor(right / self.ts, self.meta.copy())
            del obj.ts_type
            return obj            

    
class Asset(TimeSeries):
    @property
    def denominated_currency(self):
        return self._get_property_value('denominated_currency')

    @property
    def risk_currency(self):
        return self._get_property_value('risk_currency')
    
    @property
    def risk_region(self):
        return self._get_property_value('risk_region')
    
    @property
    def isin(self):
        return self._get_property_value('isin')

    def convert_currency(self, fx, to_currency, hedging_ratio,
                         hedging_frequency=DEFAULT_FX_HEDGING_FREQUENCY,
                         cross_currency=DEFAULT_CROSS_CURRENCY):
        return fx.convert_currency(self, to_currency, hedging_ratio,
                         hedging_frequency=hedging_frequency, cross_currency=cross_currency)
    
    
class Cash(Asset):
    pass


class CommoditySpot(Asset):
    """ A commodity spot Security TimeSeries. """
    @property
    def sector(self):
        return self._get_property_value('sector')
    
    @property
    def sub_sector(self):
        return self._get_property_value('sub_sector')


class Strategy(Asset):
    pass


class Equity(Asset):
    @property
    def issuer_name(self):
        return self._get_property_value('issuer_name')

        
class CommonStock(Equity):
    @property
    def domicile(self):
        return self._get_property_value('domicile')

    @property
    def risk_region(self):
        return self.domicile

    @property
    def sector(self):
        return self._get_property_value('sector')
    
    @property
    def industry_group(self):
        return self._get_property_value('industry_group')
    
    @property
    def industry(self):
        return self._get_property_value('industry')
    
    @property
    def sub_industry(self):
        return self._get_property_value('sub_industry')
    
    
class PreferredStock(Equity):
    pass


class ExchangeTradedEquity(Equity):
    @property
    def underlier_sec_code(self):
        return self._get_property_value('underlier_sec_code')
    
    @property
    def leverage(self):
        return self._get_property_value('leverage', np.nan)
    
    
class ETF(ExchangeTradedEquity):
    pass

    
class ETN(ExchangeTradedEquity):
    pass


class Bond(Asset):
    @property
    def issuer_name(self):
        return self._get_property_value('issuer_name')
        
    @property
    def par_value(self):
        return self._get_property_value('par_value', np.nan)
    
    @property
    def issue_date(self):
        return self._get_property_value('issue_date')
        
    @property
    def maturity_date(self):
        return self._get_property_value('maturity_date')
        
    @property
    def day_count(self):
        return self._get_property_value('day_count')
        
    @property
    def coupon_rate(self):
        return self._get_property_value('coupon_rate', np.nan)
    
    @property
    def coupon_frequency(self):
        return self._get_property_value('coupon_frequency', np.nan)


class StraightBond(Bond):
    pass


class FloatinRateNote(Bond):
    pass


class OriginalIssueDiscount(Bond):
    pass


class InflationProtectedSecurity(Bond):
    pass


class Derivative(Asset):
    @property
    def expiration_date(self):
        return self._get_property_value('expiration_date')


class Factor(TimeSeries):
    pass


class Forward(Derivative):
    pass


class Future(Derivative):
    pass    
   

class Option(Derivative):
    @property
    def strike(self):
        return self._get_property_value('strike', np.nan)

    @property
    def exercise_type(self):
        """ e.g. American / European / Bermudan """
        return self._get_property_value('exercise_type')

    @property
    def option_type(self):
        """ e.g. Put / Call """        
        return self._get_property_value('option_type')


class GenericIndex(TimeSeries):
    @property
    def index_provider(self):
        return self._get_property_value('index_provider')


class PriceIndex(GenericIndex):
    @property
    def risk_currency(self):
        return self._get_property_value('risk_currency')

    @property
    def risk_region(self):
        return self._get_property_value('risk_region')

    def convert_currency(self, fx, to_currency, hedging_ratio,
                         hedging_frequency=DEFAULT_FX_HEDGING_FREQUENCY,
                         cross_currency=DEFAULT_CROSS_CURRENCY):
        return fx.convert_currency(self, to_currency, hedging_ratio,
                         hedging_frequency=hedging_frequency, cross_currency=cross_currency)


class EconomicPriceIndex(PriceIndex):
    @property
    def seasonal_adjustment(self):
        return self._get_property_value('seasonal_adjustment')

    @property
    def price_type(self):
        return self._get_property_value('price_type')


class AssetIndex(PriceIndex):
    @property
    def denominated_currency(self):
        return self._get_property_value('denominated_currency')

    @property
    def risk_currency(self):
        return self._get_property_value('risk_currency')

    @property
    def risk_region(self):
        return self._get_property_value('risk_region')
    
    @property
    def hedging_ratio(self):
        return self._get_property_value('hedging_ratio', 0)


class BondIndex(AssetIndex):
    @property
    def issuer_segment(self):
        return self._get_property_value('issuer_segment')
    
    @property
    def ratings_segment(self):
        return self._get_property_value('ratings_segment')

    @property
    def maturity_bucket(self):
        return self._get_property_value('maturity_bucket')
    
    @property
    def inflation_protected(self):
        return self._get_property_value('inflation_protected', False)


class EquityIndex(AssetIndex):
    @property
    def market_cap(self):
        return self._get_property_value('market_cap')

    @property
    def factor_style(self):
        return self._get_property_value('factor_style')

    @property
    def gics_sector(self):
        return self._get_property_value('gics_sector')


class CommodityIndex(AssetIndex):
    @property
    def sector(self):
        return self._get_property_value('sector')

    @property
    def sub_sector(self):
        return self._get_property_value('sub_sector')


class RealEstateIndex(AssetIndex):
    @property
    def segment(self):
        return self._get_property_value('segment')


class HedgeFundIndex(AssetIndex):
    @property
    def strategy(self):
        return self._get_property_value('strategy')

    @property
    def substrategy(self):
        return self._get_property_value('substrategy')

    @property
    def weighting(self):
        return self._get_property_value('weighting')


class Rates(TimeSeries):
    @property
    def tenor(self):
        return self._get_property_value('tenor')
    
    @property
    def tenor_in_years(self):    
        return self._get_property_value('tenor_in_years', np.nan)

    @property
    def index_provider(self):
        return self._get_property_value('index_provider')


class FX(Rates):    
    @property
    def base_currency(self):
        return self._get_property_value('base_currency')

    @property
    def quote_currency(self):
        return self._get_property_value('quote_currency')
    
    @property
    def currency_pair(self):
        return self._get_property_value('currency_pair')

    def get_fx_spot_rates(self, target_ccy_pairs, cross_currency=DEFAULT_CROSS_CURRENCY):
        """ Get the FX spot rates. """
        return self.get_fx_rates(target_ccy_pairs, target_tenors=TENOR_SPOT, 
                                 cross_currency=cross_currency)
    
    def get_fx_rates(self, 
                     target_ccy_pairs, 
                     target_tenors=TENOR_SPOT,
                     cross_currency=DEFAULT_CROSS_CURRENCY):
        """ Function to retrieve FX time series for target currency pairs and tenors.
            Arguments:
                target_ccy_pairs: a list or string of the currency pair(s) for which data is required.
                target_tenors: a list or string of the tenor(s) for which data is required. 
                               Default tenor is the 'spot' value.
                cross_currency: if no exchange rate is found corresponding to a target currency pair, then
                          the code will try to construct it by going through the cross currency.
                          For example, to make EUR/GBP, we can go through USD with EUR/USD and USD/GBP
            
            Returns:
                A pandas DataFrame, with columns dual-indexed by the target currency pairs AND the target tenors.
        """
        if len(set(self.series_type_code)) > 1:
            raise NotImplementedError('The subtle algorithm needed to handle different series types has not been implemented.')

        if not np.all(self.ts_type == 'levels'):
            raise NotImplementedError('The algorithm needed to handle different time series types has not been implemented.')

        # Get the correctly formatted tenor and currency pair information from the input arguments
        target_ccy_pairs, target_tenors = pyfintools.security.helper.format_ccy_pair_and_tenor_info(
                                                                target_ccy_pairs, target_tenors)
            
        # Create an instance of the FXHelper to assist with exchange rate construction
        fx_helper = pyfintools.security.helper.FXHelper(base_currency=self.base_currency,
                                                   quote_currency=self.quote_currency,
                                                   labels=self.columns,
                                                   tenor=self.tenor,
                                                   cross_currency=cross_currency)
        instructions, req_labels = fx_helper.get_ccy_instructions(target_ccy_pairs, target_tenors)

        # Use the helper object to construct the exchange rates with the target tenors
        fx_ts = fx_helper.create_exchange_rates_from_instructions(instructions, self.ts)
        
        # Get the meta data
        metadata = fx_helper.get_metadata_from_instructions(instructions, self.meta, target_ccy_pairs, target_tenors)
        fx_ts.columns = metadata.columns
        
        # Create a new TimeSeries object
        return from_metadata(fx_ts, metadata)

    def _check_data_for_currency_conversion(self, asset, to_currency):
        assert isinstance(to_currency, str), "Argument 'to_currency' must be a string."
        
        if asset.shape[1] and len(set(asset.ts_type)) != 1:
            raise NotImplementedError('Currency conversion unsupported when ts_type is not unique for input assets.')
        
        if self.shape[1] and len(set(self.ts_type)) != 1:
            raise NotImplementedError('Currency conversion unsupported when ts_type is not unique for the FX panel.')
        
    def _format_data_for_currency_conversion(self, to_currency, hedging_ratio, asset_denom_ccys, asset_risk_ccys):
        if not isinstance(hedging_ratio, Iterable):
            target_hedging_ratio = np.array([hedging_ratio] * asset_denom_ccys.size, dtype=float)
        else:
            target_hedging_ratio = np.array(hedging_ratio, dtype=float)
        
        if not np.all(np.isclose(0, target_hedging_ratio) | np.isclose(1, target_hedging_ratio)):
            raise NotImplementedError('FX conversion with partial hedging ratios is not currently implemented.')
        else:
            target_hedging_ratio = target_hedging_ratio.astype(int)
        return target_hedging_ratio

    def _find_currency_conversion_methods(self, to_currency, target_hedging_ratio, 
                                          asset_names, asset_denom_ccys, asset_risk_ccys, asset_hdg_ratios):
        """ Determine the type of currency conversion computation that is required, and provide
            the method name.
        """
        methods = defaultdict(list)
        for j, name in enumerate(asset_names):
            if not np.isclose(asset_hdg_ratios[j], 0) and not np.isclose(asset_hdg_ratios[j], 1):
                methods[_HDG_PARTIAL].append(name)
            elif not np.isclose(target_hedging_ratio[j], 0) and not np.isclose(target_hedging_ratio[j], 1):
                methods[_HDG_PARTIAL].append(name)
            elif asset_denom_ccys[j] == to_currency:
                if asset_risk_ccys[j] == to_currency:
                    methods[_COMPLETE].append(name)
                elif np.isclose(target_hedging_ratio[j], asset_hdg_ratios[j]):
                    methods[_COMPLETE].append(name)
                elif np.isclose(0, asset_hdg_ratios[j]) and np.isclose(1, target_hedging_ratio[j]):
                    methods[_UNH_TO_HDG].append(name)
                elif np.isclose(1, asset_hdg_ratios[j]) and np.isclose(0, target_hedging_ratio[j]):
                    methods[_HDG_TO_UNH].append(name)
                else:
                    methods[_UNKNOWN].append(name)
            elif asset_risk_ccys[j] == to_currency:
                if np.isclose(0, asset_hdg_ratios[j]):
                    methods[_UNH_TO_UNH].append(name)
                elif np.isclose(1, asset_hdg_ratios[j]):
                    methods[_HDG_TO_UNH].append(name)
                else:
                    methods[_UNKNOWN].append(name)
            elif np.isclose(1, asset_hdg_ratios[j]) and np.isclose(1, target_hedging_ratio[j]):
                methods[_HDG_TO_HDG].append(name)
            elif np.isclose(0, asset_hdg_ratios[j]) and np.isclose(1, target_hedging_ratio[j]):
                methods[_UNH_TO_HDG].append(name)
            elif np.isclose(0, asset_hdg_ratios[j]) and np.isclose(0, target_hedging_ratio[j]):
                methods[_UNH_TO_UNH].append(name)
            elif np.isclose(1, asset_hdg_ratios[j]) and np.isclose(0, target_hedging_ratio[j]):
                methods[_HDG_TO_UNH].append(name)
            else:
                methods[_UNKNOWN].append(name)
        return methods

    def convert_currency(self,
                         asset,
                         to_currency,
                         hedging_ratio,
                         hedging_frequency=DEFAULT_FX_HEDGING_FREQUENCY,
                         cross_currency=DEFAULT_CROSS_CURRENCY):
        """ Convert the currency of an Asset TimeSeries object.
            
            Arguments:
                asset: (Asset) a TimeSeries object that inherits from Asset
                to_currency: (str) the currency to which we want to convert the time series
                hedging_ratio: (float) the hedging ratio with which we want to perform
                    the currency conversion
                hedging_frequency: (str) the frequency with which we want to rebalance
                    the currency hedge. This input is a single character, corresponding to
                    the frequency conventions used by the pandas library. Currently, 
                    only monthly currency hedging ('M') is supported.
                cross_currency: (str) the currency through which we will pass if the desired
                    currency pair is not directly available. Default is USD. 
                    For example, if we need to get AUD/EUR but have only AUD/USD and EUR/USD, then
                    having cross_currency == 'USD' will allow this function to correctly obtain
                    AUD/EUR using our existing data.
        """
        if not asset.ts.size:
            return asset
        else:
            self._check_data_for_currency_conversion(asset, to_currency)
        
        # Perform data checks and format input data
        target_hedging_ratio = self._format_data_for_currency_conversion(to_currency, hedging_ratio, 
                                                                         asset_denom_ccys=asset.denominated_currency,
                                                                         asset_risk_ccys=asset.risk_currency)
        
        # Get the conversion methods required across the different assets based on hedging ratios and currencies
        if not 'hedging_ratio' in asset.meta.index:
            asset_hdg_ratios = [0] * asset.ts.shape[1]
        else:
            asset_hdg_ratios = asset.hedging_ratio        
        conv_methods = self._find_currency_conversion_methods(to_currency, 
                                                              target_hedging_ratio,
                                                              asset_names=asset.columns, 
                                                              asset_denom_ccys=asset.denominated_currency,
                                                              asset_risk_ccys=asset.risk_currency,
                                                              asset_hdg_ratios=asset_hdg_ratios)
        
        if _UNKNOWN in conv_methods:
            raise ValueError('Unknown FX conversion methods for {}'.format(conv_methods[_UNKNOWN]))
        if _HDG_PARTIAL in conv_methods:
            raise NotImplementedError('Partial hedging not supported for {}'.format(conv_methods[_HDG_PARTIAL]))
        if _HDG_TO_HDG in conv_methods:
            raise NotImplementedError('Converting hedged to hedged time series is not supported for {}'.format( \
                                                                                       conv_methods[_HDG_TO_HDG]))
        if _HDG_TO_UNH in conv_methods:
            raise NotImplementedError('Converting hedged to unhedged time series is not supported for {}'.format( \
                                                                                       conv_methods[_HDG_TO_UNH]))

        # Get the time series that do not need any further processing
        if _COMPLETE in conv_methods:
            complete_ts = asset[conv_methods[_COMPLETE]].copy()
        else:
            complete_ts = asset[[]]  # Create an empty asset object

        # Get the time series that will be converted unhedged
        if _UNH_TO_UNH in conv_methods:
            ts_for_unhedging = asset[conv_methods[_UNH_TO_UNH]].copy()
            ts_unh = self._convert_currency_unhedged_to_unhedged(ts_for_unhedging, to_currency, cross_currency=cross_currency)
        else:
            ts_unh = asset[[]]  # Create an empty asset object

        # Get the time series that will be converted unhedged
        if _UNH_TO_HDG in conv_methods:
            ts_for_hedging = asset[conv_methods[_UNH_TO_HDG]].copy()
            ts_hdg = self._convert_currency_unhedged_to_hedged(ts_for_hedging, to_currency,
                                                               cross_currency=cross_currency, 
                                                               hedging_frequency=hedging_frequency)
        else:
            ts_hdg = asset[[]]  # Create an empty asset object            

        # Join the time series and return the columns in the same order as the input
        asset_ts = complete_ts.merge([ts_unh, ts_hdg])
        return asset_ts[asset.columns]

    def _convert_currency_unhedged_to_hedged(self, 
                                             asset, 
                                             to_currency, 
                                             hedging_frequency=DEFAULT_FX_HEDGING_FREQUENCY,
                                             cross_currency=DEFAULT_CROSS_CURRENCY):
        """ Convert from an unhedged time series to a hedged time series. """
        # Save the original ts_type information so we can convert back at the end
        orig_ts_type = asset.ts_type[0]
        
        # First, calculate the unhedged levels
        asset_levels = asset.to_levels()
        unh_asset_levels = self.convert_currency(asset,
                                                 to_currency=to_currency,
                                                 hedging_ratio=0,
                                                 cross_currency=cross_currency)

        # Get the proceeds from hedging from the different currencies
        unh_asset_rtns = unh_asset_levels.to_simple_returns()
        hedging_proceeds, _ = self.get_hedging_proceeds(to_currency=to_currency,
                                                        from_currency=unh_asset_rtns.risk_currency,
                                                        hedging_frequency=hedging_frequency,
                                                        cross_currency=cross_currency)

        # Check that none of the hedging proceeds have all NaN's as entries
        is_ts_nan = np.all(np.isnan(hedging_proceeds.values), axis=0)
        if any(is_ts_nan):
            raise ValueError('Some hedging proceeds are not available: {}'.format(\
                                    hedging_proceeds.columns.values[is_ts_nan]))
            
        # Combine the unhedged returns and the hedging proceeds
        unh_asset_rtns, hedging_proceeds = unh_asset_rtns.align(hedging_proceeds, join='inner', axis=0)
        ccy_pairs = [f'{risk_ccy}/{base_ccy}' for risk_ccy, base_ccy in \
                             zip(unh_asset_rtns.risk_currency, unh_asset_rtns.denominated_currency)]
        hdg_rtns = unh_asset_rtns + hedging_proceeds.ts[ccy_pairs].values
        hdg_rtns.ts_type = pyfintools.tools.fts.TS_TYPE_SIMPLE_RETURNS

        # Return the unhedged time series with the original ts_type, and update the meta data
        asset_ts = hdg_rtns.to_ts_type(orig_ts_type)        
        asset_ts.meta.loc['denominated_currency'] = to_currency
        asset_ts.meta.loc['hedging_ratio'] = 1
        return asset_ts

    def _convert_currency_unhedged_to_unhedged(self, 
                                               asset, 
                                               to_currency, 
                                               cross_currency=DEFAULT_CROSS_CURRENCY):
        """ Convert from unhedged time series to another unhedged time series in a different currency. """
        # Make sure that we have implemented the methodology necessary to do the currency conversion
        orig_cols = asset.columns        
        idx_done = self._is_hedged_to_target(asset, to_currency, target_hedging_ratio=0)
        if np.any(idx_done):
            asset_ts = asset[asset.columns[idx_done]]
            asset = asset[asset.columns[~idx_done]]
        else:
            asset_ts = None
        
        # Save the original ts_type information so we can convert back at the end
        orig_ts_type = asset.ts_type[0]

        # First, calculate the unhedged levels
        asset_levels = asset.to_levels()        
        unh_asset_levels = self._convert_currency_unhedged_to_unhedged_levels(asset_levels,
                                                                              to_currency=to_currency,
                                                                              cross_currency=cross_currency)

        # Return the unhedged time series with the original ts_type
        if asset_ts is not None:
            asset_ts = asset_ts.merge(unh_asset_levels.to_ts_type(orig_ts_type))
        else:
            asset_ts = unh_asset_levels.to_ts_type(orig_ts_type)

        asset_ts.meta.loc['denominated_currency'] = to_currency
        asset_ts.meta.loc['hedging_ratio'] = 0
        return asset_ts[orig_cols]

    def _is_hedged_to_target(self, asset, to_currency, target_hedging_ratio):
        """ Check if the asset input is hedged to its target hedging ratio. 
        
            This method examines the meta data on the 'asset' input and evaluates 
            whether or not it is hedged according to the desired inputs: 'to_currency' 
            and 'target_hedging_ratio'
        """
        if target_hedging_ratio == 1:
            # Make sure that we have implemented the methodology necessary to do the currency conversion
            if hasattr(asset, 'hedging_ratio'):
                current_hedging_ratio = np.array(asset.hedging_ratio, dtype=float)
            else:
                current_hedging_ratio = np.zeros((asset.ts.shape[1],), dtype=float)

            if not np.all(np.isclose(0, current_hedging_ratio) | \
                         (np.isclose(1, current_hedging_ratio) & (to_currency == np.array(asset.denominated_currency)))):
                raise NotImplementedError('FX conversion for time series that are already hedged is not yet supported.')
            
            return (to_currency == asset.denominated_currency) & \
                   (np.isclose(1, current_hedging_ratio) | (asset.risk_currency == asset.denominated_currency))
        elif target_hedging_ratio == 0:
            if hasattr(asset, 'hedging_ratio'):
                current_hedging_ratio = np.array(asset.hedging_ratio, dtype=float)
            else:
                current_hedging_ratio = np.zeros((asset.ts.shape[1],), dtype=float)

            if not np.all(np.isclose(0, current_hedging_ratio)):
                raise NotImplementedError('FX conversion for time series that are already hedged is not yet supported.')

            return (to_currency == asset.denominated_currency) & \
                   (np.isclose(0, current_hedging_ratio) | (asset.risk_currency == asset.denominated_currency))            
        else:
            raise ValueError(f'Unsupported target hedging ratio: {target_hedging_ratio}')
        
    def _convert_currency_unhedged_to_unhedged_levels(self, asset, to_currency,
                                                      cross_currency=DEFAULT_CROSS_CURRENCY):
        """ Convert the currency from unhedged levels (prices) to unhedged levels in a new currency.
        """
        if not np.all(asset.ts_type == 'levels'):
            raise ValueError('Not supported unless all columns have ts_type == "levels".')

        # Get the Spot FX rates needed for currency conversion
        ccy_pairs = [f'{bc}/{to_currency}' for bc in asset.denominated_currency]
        spot_rates = self.get_fx_spot_rates(list(set(ccy_pairs)), cross_currency=cross_currency).to_levels()
        spot_rates.columns = spot_rates.currency_pair
        
        # Align the FX and asset time series
        asset, spot_rates = asset.align(spot_rates, join='outer', axis=0)

        # Check that none of the FX rates have all NaN's as entries
        is_ts_nan = np.all(np.isnan(spot_rates.values), axis=0)
        if any(is_ts_nan):
            raise ValueError('Some exchange rates are not available: {}'.format(ccy_pairs[is_ts_nan]))

        # Ensure that the identity exchange rate is not ever NaN
        identity_ccy_pair = f'{to_currency}/{to_currency}'
        spot_rates.ts.iloc[:,spot_rates.columns == identity_ccy_pair] = 1.

        # Create a new TimeSeries object 
        _ts = asset.ts * spot_rates.ts[ccy_pairs].values
        _meta = asset.meta.copy()
        _meta.loc['denominated_currency'] = to_currency        
        return asset._constructor(_ts, _meta)

    def get_hedging_proceeds(self, 
                             to_currency, 
                             from_currency, 
                             hedging_frequency=DEFAULT_FX_HEDGING_FREQUENCY, 
                             cross_currency=DEFAULT_CROSS_CURRENCY):
        """ Compute the difference between the hedged and unhedged time series.
        
            The difference between the hedged and unhedged time series is defined as the 'hedging proceeds.'
            These hedging proceeds can be derived by performing a strategy of rolling the FX forwards.
        """ 
        return self.get_fx_forward_returns(long_currency=to_currency,
                                           short_currency=from_currency,
                                           roll_frequency=hedging_frequency,
                                           cross_currency=cross_currency)

    def get_fx_forward_returns(self, 
                               long_currency, 
                               short_currency, 
                               roll_frequency=DEFAULT_FX_HEDGING_FREQUENCY,
                               cross_currency=DEFAULT_CROSS_CURRENCY):
        """ Compute the returns of a strategy of rolling FX forwads.
        
            Arguments:
                long_currency: (str) the currency of the long leg of the contract (e.g. AUD)
                short_currency: (str) the currency of the short/base leg of the contract (e.g. USD)
                roll_frequency: (str) how often the contracts are rolled/rebalanced
                cross_currency: (str) the currency through which we cross if a target currency pair is not
                    available. For example, using USD as a cross currency to obtain AUD/EUR from USD/AUD and USD/EUR.
        """
        if roll_frequency.lower() != '1m':
            raise NotImplementedError('Rolling forwards is currently only supported for 1-month roll frequency.')
        else:
            required_fwd_tenors = [roll_frequency]

        # Get the required currency pairs
        currency_pairs = pyfintools.security.helper.get_currency_pairs_from_inputs(long_currency, short_currency)
        uniq_currency_pairs = list(set(currency_pairs))
        
        # Get time series for the spot exchange rates
        fx_rates = self.get_fx_rates(uniq_currency_pairs, 
                                     target_tenors=required_fwd_tenors + [TENOR_SPOT],
                                     cross_currency=cross_currency)

        # Remove rows where neither spots or forwards exist (ignoring identity pairs like USD/USD) 
        identity_pair = [cp[:3] == cp[-3:] for cp in fx_rates.currency_pair]
        good_vals = ~np.isnan(fx_rates.values[:,fx_rates.currency_pair != identity_pair])
        idx = fx_rates.ts.iloc[np.any(good_vals, 1)].index

        # Get the spot and forward rates
        spot_rates = fx_rates.apply_filter('tenor', TENOR_SPOT).loc[idx]
        fwd_rates = fx_rates.drop('tenor', TENOR_SPOT).loc[idx]
        
        # Put the columns in the same order as the input arguments
        spot_rates = spot_rates.select('currency_pair', uniq_currency_pairs)
        fwd_rates = fwd_rates.select('currency_pair', uniq_currency_pairs)        

        # Get rid of the multi-indexed columns, and just use the currency pair as the column name
        spot_rates.columns = spot_rates.currency_pair
        fwd_rates.columns = fwd_rates.currency_pair

        if pd.infer_freq(spot_rates.index) not in pyfintools.tools.freq.MONTHLY_FREQUENCIES:
            raise NotImplementedError('Hedging is currently only supported for time series with monthly frequencies.')

        # Calculate the return of an FX forward or swap
        spot_rtns = spot_rates.to_simple_returns()
        carry = -1 + fwd_rates.shift(1) / spot_rates.shift(1).values
        hedging_proceeds = -spot_rtns + carry.values
        
        # Change the 'ts_type' before returning the objects
        carry.ts_type = pyfintools.tools.fts.TS_TYPE_RATES
        hedging_proceeds.ts_type = pyfintools.tools.fts.TS_TYPE_SIMPLE_RETURNS
        return hedging_proceeds, carry


class PPP(FX):
    """ A class derived from the FX class that allows construction of PPP time series from inflation data.
    
        The main methods on this class are class methods that provide different methods for deriving PPP
        series from inflation.
    """

    @classmethod
    def from_prices(cls, prices, fx_spot_rates, reference_region='US', method='ema', window=None, min_periods=1):
        """ Create a PPP TimeSeries object from inflation data.
        
            Arguments:
                prices: (TimeSeries) the PPI/CPI data that will be used to create the PPP
                fx_spot_rates: (FX) the FX TimeSeries object whose values are needed to calibrate the fair value
                reference_region: (str) the region whose PPI/CPI values will be used as a reference against 
                    which all currencies are measured.
                method: (str) one of the following options:
                    in_sample: The calibration of the fair value of PPP is performed one time, and uses all of the
                        data in the time series. The result is that the fair value level at the beginning of the
                        observation period uses all of the data until the end of the period, and is therefore
                        not an out-of-sample fit (and is hence in-sample)
                    ema: use an out-of-sample calibration, with an Exponential Moving Average (EMA) whose
                        window is defined via the 'window' argument
                    expanding: use an out-of-sample calibration with an expanding window
                    rolling: use an out-of-sample calibration with a rolling window
                window: (str/int) for out-of-sample methods, this specifies how large of a window should be used
                    in the calibration. The 'window' argument can either be an integer which explicitly tells us
                    how many observations to use in each window. 
                    Or, it can be a frequency string like '12M', '3M', '90D', etc.
                min_periods: (int) the minimum number of periods to include. This is currently only used when
                    the method is set to 'expanding'.
                
        """
        # Get the unscaled PPP time series
        unscaled_ppp_ts = cls._from_prices_unscaled(prices, reference_region=reference_region)

        # Make sure that the FX spot rates are using the same currency pairs as the PPP time series        
        unscaled_ppp_ts = unscaled_ppp_ts.get_fx_spot_rates(fx_spot_rates.currency_pair.values)
        
        # Make sure the columns are equal to the currency pairs
        unscaled_ppp_ts.columns = unscaled_ppp_ts.currency_pair.values
        
        if 'in_sample' == method:
            return cls._from_prices_in_sample(unscaled_ppp_ts, fx_spot_rates, reference_region=reference_region)
        else:
            return cls._from_prices_out_of_sample(unscaled_ppp_ts, fx_spot_rates, reference_region=reference_region,
                                                  method=method, window=window, min_periods=min_periods)

    @classmethod
    def _from_prices_out_of_sample(cls, unscaled_ppp_ts, fx_spot_rates, reference_region='US', method='ema',
                                   window=None, min_periods=1):
        rescale_factors = []
        for ccy_pair in unscaled_ppp_ts.currency_pair.values:
            fx = fx_spot_rates.select('currency_pair', ccy_pair).ts.dropna()
            unscl_ppp = unscaled_ppp_ts.select('currency_pair', ccy_pair).ts.dropna()
            fx, unscl_ppp = fx.align(unscl_ppp, join='inner')
            
            if 'ema' == method:
                w = pyfintools.tools.freq.extract_window_size(window, unscaled_ppp_ts)
                df = np.log(fx) - np.log(unscl_ppp)
                const = talib.EMA(df.iloc[:,0], w)
            elif 'expanding' == method:
                const = np.log(fx).expanding(min_periods=min_periods).mean() - \
                        np.log(unscl_ppp).expanding(min_periods=min_periods).mean()
            elif 'rolling' == method:
                w = pyfintools.tools.freq.extract_window_size(window, unscaled_ppp_ts)                
                const = np.log(fx).rolling(w).mean() - \
                        np.log(unscl_ppp).rolling(w).mean()
            else:
                raise ValueError(f'Unsupported method type: {method}')

            rsc_fac = np.exp(const)                
            rescale_factors.append(rsc_fac)

        rescale_ts = pd.concat(rescale_factors, axis=1)
        ppp_OS_ts = unscaled_ppp_ts.loc[rescale_ts.index] * rescale_ts.values
        return ppp_OS_ts
        
    @classmethod
    def _from_prices_in_sample(cls, unscaled_ppp_ts, fx_spot_rates, reference_region='US'):
        rescale_factors = []
        for ccy_pair in unscaled_ppp_ts.currency_pair.values:
            fx = fx_spot_rates.select('currency_pair', ccy_pair).ts.dropna()
            unscl_ppp = unscaled_ppp_ts.select('currency_pair', ccy_pair).ts.dropna()
            fx, unscl_ppp = fx.align(unscl_ppp, join='inner')

            const = np.log(fx).mean() - np.log(unscl_ppp).mean()
            rsc_fac = np.exp(const)
            rescale_factors.append(rsc_fac.values[0])

        ppp_IS_ts = unscaled_ppp_ts * np.array(rescale_factors, dtype=float)
        return ppp_IS_ts
        
    @classmethod
    def _from_prices_unscaled(cls, prices, reference_region='US'):
        # Get the reference prices and currency
        ref_price = prices.select('risk_region', reference_region)
        ref_ccy = ref_price.risk_currency.values[0]

        # Construct the unscaled PPP time series
        unscaled_ppp_ts = prices / ref_price.values
        unscaled_ppp_ts = unscaled_ppp_ts.drop('risk_region', reference_region)

        # Get the meta data
        meta = pd.DataFrame.from_dict(
                               dict(sec_code=unscaled_ppp_ts.sec_code,
                                    ts_type='levels',
                                    category_1_code='FX',
                                    category_2_code='PPP',
                                    tenor=pyfintools.security.constants.TENOR_SPOT,
                                    tenor_in_years=0.0,
                                    base_currency=ref_ccy,
                                    quote_currency=unscaled_ppp_ts.risk_currency,
                                    currency_pair=[f'{ref_ccy}/{ccy}' for ccy in unscaled_ppp_ts.risk_currency],
                                    index_provider=unscaled_ppp_ts.index_provider)).T

        ts = cls(unscaled_ppp_ts.ts, meta)
        ts.columns = ts.currency_pair
        return ts


class InterestRate(Rates):
    """ A class that contains time series and meta data information for interest rates. 
    """
    @property
    def denominated_currency(self):
        return self._get_property_value('denominated_currency')

    @property
    def risk_currency(self):
        return self._get_property_value('risk_currency')

    @property
    def risk_region(self):
        return self._get_property_value('risk_region')

    @property
    def issuer_name(self):
        return self._get_property_value('issuer_name')
    
    @property
    def issuer_segment(self):
        return self._get_property_value('issuer_segment')

    @property
    def ratings_segment(self):
        return self._get_property_value('ratings_segment')

    @property
    def rate_type(self):
        return self._get_property_value('rate_type')
    
    @property
    def inflation_protected(self):
        return self._get_property_value('inflation_protected', False)    


class YieldCurve(InterestRate):
    """ A class that allows manipulations with yield curve data.
    """
    def __init__(self, ts, meta):
        super(YieldCurve, self).__init__(ts, meta)
        
        if hasattr(meta, 'denominated_currency'):
            denom_ccys = meta.denominated_currency
            assert len(set(denom_ccys)) <= 1, 'Denominated currency must be unique for a yield curve object.'

        if hasattr(meta, 'risk_currency'):
            risk_ccys = meta.risk_currency
            assert len(set(risk_ccys)) <= 1, 'Risk currency must be unique for a yield curve object.'

        # Sort by tenor_in_years
        meta = meta.T.sort_values('tenor_in_years').T
        ts = ts[meta.columns]
            
        self.ts = ts
        self.meta = meta
        
        # Cache some calculated properties
        self._curve_type = None
        self._compounding_freq = None        

    @property
    def curve_type(self):
        """ Check to see if there is a unique 'rate_type' for the underlying interest rates with tenors > 1y
            E.g., is this a zero, par, or forward curve. """
        if self._curve_type is None:
            rate_types = list(set(self.meta.T.query("tenor_in_years > 1").T.loc['rate_type']))
            if len(rate_types) != 1:
                self._curve_type = ""
            else:
                self._curve_type = rate_types[0]
        return self._curve_type
    
    @property
    def compounding_freq(self):
        """ Check to see if there is a unique 'compounding_freq' for the underlying interest rates with tenors > 1y
            E.g., are these continuously compounded rates, or is the compounding frequency 2 (times per year). """        
        if self._compounding_freq is None:
            cmpd_freqs = list(set(self.meta.T.query("tenor_in_years > 1").T.loc['compounding_freq']))
            if len(cmpd_freqs) != 1:
                # If there is no agreement amoung the compounding frequency, then return NaN as a signal something is wrong
                self._compounding_freq = np.nan
            else:
                # Otherwise, use the unique frequency, if it is an integer
                if not pyfintools.tools.utils.is_integer(cmpd_freqs[0]):
                    raise ValueError('Valid compounding frequencies must be integers: {}'.format(cmpd_freqs[0]))
                else:
                    self._compounding_freq = int(cmpd_freqs[0])
        return self._compounding_freq

    def get_yields(self, target_tenors, pricing_dates=None, extrap='left'):
        """ Get a time series of yields for a set of tenors, using spline interpolation as necessary. 
        
            Arguments:
                target_tenors: can be a single tenor, a list/numpy array of tenors, a pandas Series/DataFrame, 
                       or a TimeSeries object. If a time series is provided, then the target tenors can change
                       over time. Otherwise, the returned value is in reference to the fixed tenor(s).
                extrap: Whether or not to 'extrapolate' values outside the range of available tenors.
                        Supported values are "left", "right", "both", "neither". 
                        This extrapolation method just uses the closest available tenor, and does not use any
                             linear or other complex extrapolation methods.
                pricing_dates: the dates on which to obtain yields
        """
        if pyfintools.tools.utils.is_numeric(target_tenors):
            target_tenors = np.array([target_tenors], dtype=float)
        elif isinstance(target_tenors, TimeSeries):
            target_tenors = target_tenors.ts
        elif isinstance(target_tenors, pd.Series):
            target_tenors = pd.DataFrame(target_tenors)
        else:
            target_tenors = np.array(list(target_tenors), dtype=float)

        if isinstance(target_tenors, pd.DataFrame):
            target_index = target_tenors.index
            n_cols = target_tenors.shape[1]
        else:
            target_index = self.index
            n_cols = target_tenors.size

        # Only use the pricing dates if they are provided
        if pricing_dates is not None:
            target_index = pd.DatetimeIndex(pricing_dates)

        tol = 1e-4            
        x = self.tenor_in_years.values
        interp_vals = np.nan * np.zeros((target_index.size, n_cols), dtype=np.float32)
        for t, idx in enumerate(target_index):
            if idx in self.index:
                # Only use non-NaN values for interpolation
                y = self.ts.loc[idx].values
                idx_good = ~np.isnan(y)
                X = x[idx_good]
                Y = y[idx_good]

                # Using s=0 ensures that the spline goes through all data points
                # To perform a cubic spline, we need at least 3 data points
                if len(X) > 3:
                    vals_t = scipy.interpolate.splrep(X, Y, s=0)

                    # Make sure that we ignore extrapolated values outside of the range of available data points
                    if isinstance(target_tenors, pd.DataFrame):
                        tenor_vals = target_tenors.values[t,:]
                    else:
                        tenor_vals = target_tenors

                    idx_interp = np.array([tnr > min(X) - tol and tnr < max(X) + tol for tnr in tenor_vals], dtype=bool)
                    if np.any(idx_interp):
                        interp_vals[t, idx_interp] = scipy.interpolate.splev(tenor_vals[idx_interp], vals_t, der=0)

                    # Perform any extrapolation if desired. Here we hold missing values constant
                    if extrap == 'left' or extrap == 'both':
                        interp_vals[t, tenor_vals <= X[0]] = Y[0]

                    if extrap == 'right' or extrap == 'both':
                        interp_vals[t, tenor_vals >= X[-1]] = Y[-1]

        if isinstance(target_tenors, pd.DataFrame):
            df = pd.DataFrame(interp_vals, index=target_index, columns=target_tenors.columns)
        else:
            df = pd.DataFrame(interp_vals, index=target_index, columns=target_tenors)
        return df
    
    def get_yield_curve(self, target_tenors, pricing_dates=None, extrap='left'):
        """ Get a new YieldCurve object, possibly with a different set of tenors or pricing dates. 
        
            Arguments:
                target_tenors: can be a single tenor, a list/numpy array of tenors, a pandas Series/DataFrame, 
                       or a TimeSeries object. If a time series is provided, then the target tenors can change
                       over time. Otherwise, the returned value is in reference to the fixed tenor(s).
                extrap: Whether or not to 'extrapolate' values outside the range of available tenors.
                        Supported values are "left", "right", "both", "neither". 
                        This extrapolation method just uses the closest available tenor, and does not use any
                             linear or other complex extrapolation methods.
                pricing_dates: the dates on which to obtain yields
        """
        yield_ts = self.get_yields(target_tenors=target_tenors, pricing_dates=pricing_dates, extrap=extrap)
        
        # Call the classmethod to create the new YieldCurve object
        return self.__class__.from_dataframe(yield_ts, 
                                             tenor_in_years=target_tenors,
                                             risk_currency=self.risk_currency.values[0],
                                             denominated_currency=self.denominated_currency.values[0],
                                             compounding_freq=self.compounding_freq,
                                             rate_type=self.curve_type)
        
    def get_zero_rates(self, target_tenors, pricing_dates=None, extrap='left'):
        """ Get the zero/spot rates for a set of target tenors
        
            Arguments:
                target_tenors: can be a single tenor, a list/numpy array of tenors, a pandas Series/DataFrame, 
                       or a TimeSeries object. If a time series is provided, then the target tenors can change
                       over time. Otherwise, the returned value is in reference to the fixed tenor(s).
                extrap: Whether or not to 'extrapolate' values outside the range of available tenors.
                        Supported values are "left", "right", "both", "neither". 
                        This extrapolation method just uses the closest available tenor, and does not use any
                             linear or other complex extrapolation methods.
                pricing_dates: the dates on which to obtain yields

        """
        if self.curve_type == RATE_TYPE_ZERO:
            return self.get_yields(target_tenors, pricing_dates=pricing_dates, extrap=extrap)
        elif self.curve_type in (RATE_TYPE_PAR, RATE_TYPE_ZERO):
            raise NotImplementedError('Conversion to zero rates it not supported for rates of type "{}"'.format(
                                                        self.curve_type))
        else:
            raise ValueError('This YieldCurve has an unknown or non-unique rate type: "{}"'.format(self.curve_type))
    
    def calc_discounted_value(self, pricing_date, cashflows, n_years_to_cashflows):
        """ Calculate the discounted present value of a series of cashflows on a specified pricing date
        
            Arguments:
                pricing_date: (str/Timestamp/datetime) the date on which the discounted 
                    present value calculation is performed
                cashflows: (list/array of floats) a list/numpy array of cash flow amounts.
                    Must have the same length as 'n_years_to_cashflows'
                n_years_to_cashflows: (list/array of floats) a list/numpy array representing the
                    number of years for each cashflow in the 'cashflows' input.
                    Must have the same length as 'cashflows'.
        """
        # Make sure the pricing date is a pandas Timestamp
        pricing_date = pd.Timestamp(pricing_date)
        
        if not pyfintools.tools.utils.is_numeric(cashflows):
            cashflows = np.array(cashflows, dtype=float)

        if not pyfintools.tools.utils.is_numeric(n_years_to_cashflows):
            n_years_to_cashflows = np.array(n_years_to_cashflows, dtype=float)

        zero_rates = self.get_zero_rates(n_years_to_cashflows, pricing_dates=[pricing_date]).values[0,:]
        if self.compounding_freq == pyfintools.constants.CONTINUOUS_COMPOUNDING:
            discounted_vals = cashflows * np.exp(-zero_rates * n_years_to_cashflows)
        elif isinstance(self.compounding_freq, int):
            discounted_vals = cashflows / np.power(1 + zero_rates / self.compounding_freq, 
                                                   self.compounding_freq * n_years_to_cashflows)
        else:
            raise ValueError('Unsupported compounding type: "{}"'.format(self.compounding_freq))
        return discounted_vals

    @classmethod
    def from_dataframe(cls, yield_ts, tenor_in_years, rate_type, compounding_freq=pyfintools.constants.CONTINUOUS_COMPOUNDING, 
                       risk_currency='', denominated_currency=''):
        """ Create a YieldCurve object by providing the time series DataFrame and some required meta data. 
        
            Arguments:
                yield_ts: (DataFrame) a pandas DataFrame with the raw yield time series
                tenor_in_years: (list) a list of the tenor (in years) of each column of 'yield_ts'
                compounding_freq: (str) the compounding frequency for the interest rates. Default is 'continuous'
                risk_currency: (str) the risk currency for the yields. Default is ''
                denominated_currency: (str) the denominated currency for the yields. Default is ''
                
            Outputs a YieldCurve object
        """
        # Create meta data
        rates_meta = pd.DataFrame.from_dict(dict(tenor_in_years=tenor_in_years,
                                                 risk_currency=risk_currency,
                                                 denominated_currency=denominated_currency,
                                                 compounding_freq=compounding_freq,
                                                 rate_type=rate_type)).T
        rates_meta.columns = yield_ts.columns
        return cls(yield_ts, rates_meta)
    
    @classmethod
    def get_implied_path_ts(cls, yc_obj, pricing_date, data_frequency='Y',
                             adjust_for_convexity=False, yield_exp_vols=None):
        """ From a set of input yields, extract the implied yield surface into the future.
        
            Assuming the yields evolve according to the expectations hypothesis, which specifies
            that the forward rates are market participants' best forecast of where future
            yields will go.
            
            Arguments:
                yc_obj: (YieldCurve) a YieldCurve TimeSeries object
                pricing_date: (str/Timestamp/datetime) the yield curve observed on the pricing
                    date will be used to compute the implied path of yields
                data_frequency: (str) the frequency with which to provide a forecast of the
                    future yield surface (e.g. 'M' for monthly, 'Y' for annually)
                adjust_for_convexity: (bool) whether or not to adjust the implied path for 
                    convexity effects. Default is False. If True, then the function will 
                    assume that market participants are incorporating profits/losses that 
                    would be attributable to convexity, and so we adjust the observed yield
                    surface to account for this effect.
                yield_exp_vols: (list of floats) the expected volatility of the yields in each
                    column of yc_obj. This input is only required if 'adjust_for_convexity' is True,
                    as the expected volatility of yields effects the profitability of convexity.
        """
        # Get the matrix of the implied path for rates
        rate_list = yc_obj.get_implied_path(data_frequency=data_frequency,
                                            adjust_for_convexity=adjust_for_convexity, 
                                            yield_exp_vols=yield_exp_vols)
        rate_mtx = np.vstack([x.values.reshape(1,-1) for x in rate_list])
        tenor_in_years = rate_list[0].tenor_in_years
        
        # Get the future dates correspoding to the future rates
        future_dates = pd.date_range(pricing_date, periods=rate_mtx.shape[0], freq=data_frequency.upper())
        
        # Combine the future dates and rates and return a DataFrame
        implied_df = pd.DataFrame(rate_mtx, index=future_dates, columns=tenor_in_years)
        
        # Put everything together to get a YieldCurve object
        return YieldCurve.from_dataframe(implied_df, 
                                         tenor_in_years=tenor_in_years,
                                         rate_type=yc_obj.rate_type)


class MonthlyYieldCurve(YieldCurve):
    """ An optimized version of the YieldCurve class; useful if only monthly yields are required.
    
        This class does no interpolation when getting yields - this makes calculations much faster if the
        required tenors are always close to even numbers of months (e.g., 6m, 1y, 15m, 5y, 65m, etc.)
    """

    def get_yields(self, target_tenors, pricing_dates=None, extrap='right'):
        """ Get a time series of yields for a set of tenors, using spline interpolation as necessary. 
        
            Arguments:
                target_tenors: can be a single tenor, a list/numpy array of tenors, a pandas Series/DataFrame, 
                       or a TimeSeries object. If a time series is provided, then the target tenors can change
                       over time. Otherwise, the returned value is in reference to the fixed tenor(s).
                extrap: Whether or not to 'extrapolate' values outside the range of available tenors.
                        Supported values are "left", "right", "both", "neither". 
                        This extrapolation method just uses the closest available tenor, and does not use any
                             linear or other complex extrapolation methods.
                pricing_dates: the dates on which to obtain yields
        """
        
        if pyfintools.tools.utils.is_numeric(target_tenors):
            target_tenors = np.array([target_tenors], dtype=float)
        else:
            target_tenors = np.array(target_tenors, dtype=float)

        target_tenors_in_months = 12 * target_tenors
        target_tenors_in_months_rounded = np.round(target_tenors_in_months, 0)

        tol = 0.2
        if np.any(np.abs(target_tenors_in_months_rounded - target_tenors_in_months) > tol):
            raise ValueError('Some target tenors are too far away from an even number of months.')
        else:
            if pricing_dates is None:
                return self.ts.loc[:, target_tenors_in_months_rounded]
            else:
                return self.ts.loc[pricing_dates, target_tenors_in_months_rounded]


class IndexSliceHelper(object):
    """ A helper class that allows TimeSeries objects to use 'loc' and 'iloc' similar to pandas DataFrames.
    """
    def __init__(self, ts_obj, index_fun):
        super(IndexSliceHelper, self).__init__()
        self.ts_obj = ts_obj
        self.index_fun = index_fun
        
    def __getitem__(self, val):
        # Extract the index information
        if isinstance(val, tuple):
            idx_0 = val[0]
            idx_1 = val[1]
        else:
            idx_0 = val
            idx_1 = None

        # Make sure that the index is never a single integer to prevent the return value being a pandas Series
        if idx_0 is not None and not isinstance(idx_0, slice) and (isinstance(idx_0, str) or not isinstance(idx_0, Iterable)):
            idx_0 = [idx_0]
        if idx_1 is not None and not isinstance(idx_1, slice) and  (isinstance(idx_1, str) or not isinstance(idx_1, Iterable)):
            idx_1 = [idx_1]            

        # Apply 'loc' or 'iloc' to the time series
        obj = self.ts_obj
        reference = obj.ts.__getattribute__(self.index_fun)
        if idx_1 is not None:
            _ts = reference[idx_0, idx_1].copy()
        else:
            _ts = reference[idx_0].copy()

        if isinstance(_ts, pd.Series):
            _ts = pd.DataFrame(_ts)

        return obj._constructor(_ts, obj.meta[_ts.columns].copy())


# Define Factory method for creating new Security Panel objects from DataFrames
def from_metadata(ts, meta):
    """ A factory method that creates an appropriate TimeSeries object from time series and meta DataFrame inputs. 
    
        The input 'meta' can contain any fields, but must at a minimum include 'category_1_code' and 'category_2_code'.
        These are used to determine which type of TimeSeries subclass should be initialized to provide
        the most appropriate set of features for this data.
        
        The input arguments are identical to what is used by the TimeSeries base class. Please
        see the documentation for the TimeSeries base class for more information about
        the requirements of these inputs.
        
        Arguments:
            ts: (DataFrame) a pandas DataFrame containing time series data
            meta: (DataFrame) a pandas DataFrame containing meta data about the time series.
    """
    class_name = pyfintools.security.helper.get_security_panel_name(cat_1_codes=meta.loc['category_1_code'], 
                                                               cat_2_codes=meta.loc['category_2_code'],
                                                               default_class_name='TimeSeries')
    class_handle = eval(class_name)
    return class_handle(ts, meta)
