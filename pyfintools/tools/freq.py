""" Defines functions that manipulate and extract time series frequency information.

    Some common examples are calculating the number of periods/observations per year 
    from a given time series, or extracting the frequency (e.g. 'M' or 'D') of a time series object.
"""

import re
import datetime
import calendar
import numpy as np
import pandas as pd

from collections import defaultdict, Iterable

DAYS_PER_YEAR = 365.24
SECONDS_PER_DAY = 24 * 3600
SECONDS_PER_YEAR = DAYS_PER_YEAR * SECONDS_PER_DAY
TENOR_SPOT = 'spot'

MONTHLY_FREQUENCIES = ('M', 'BM', 'MS', 'BMS')

MONTH_ABBREV = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

def infer_periods_per_year(index):
    """ Infer the number of periods per year from a pandas DatetimeIndex object. """
    freq = infer_freq(index)
    return get_periods_per_year(freq)

def infer_years_per_period(index):
    """ Infer the number of years per period from a pandas DatetimeIndex object. """
    freq = infer_freq(index)
    return get_years_per_period(freq)

def get_periods_per_year(freq):
    """Get the number of periods per year for an input frequency.
       Viable frequency units are: y, q, m, w, b, d, h, t, s
       Some examples:
       >>get_periods_per_year('m')
       12.0
       >>get_periods_per_year('3m')
       4.0
       >>get_periods_per_year('4w')
       13.044285714285715
    """
    units = re.sub('[0-9\.]', '', freq).lower()
    n_str = re.sub('[a-zA-Z\-]', '', freq)
    if not n_str:
        N = 1
    else:
        N = float(n_str)

    if 's' == units:
        n_base_periods = 3600 * 24 * DAYS_PER_YEAR
    elif 't' == units:
        n_base_periods = 60 * 24 * DAYS_PER_YEAR
    elif 'h' == units:
        n_base_periods = 24 * DAYS_PER_YEAR
    elif 'd' == units:
        n_base_periods = DAYS_PER_YEAR
    elif 'b' == units:
        n_base_periods = 5/7 * DAYS_PER_YEAR
    elif 'w' == units:
        n_base_periods = DAYS_PER_YEAR / 7
    elif 'm' == units:
        n_base_periods = 12.0
    elif is_quarterly_frequency(units):
        n_base_periods = 4.0
    elif 'y' == units:
        n_base_periods = 1.0
    elif 'spot' == units:
        n_base_periods = float('inf')
    else:
        raise ValueError(f'Unsupported frequency: {freq}')
    return n_base_periods / N

def is_quarterly_frequency(freq):
    fl = freq.lower()
    return 'q' == fl or fl in ['q-' + m for m in MONTH_ABBREV]

def get_years_per_period(freq):
    return 1 / get_periods_per_year(freq)

def infer_freq(index, allow_missing=False):
    """ Generalizes pandas 'infer_freq' function, which infers the frequency of a set of dates/times
    
        This function first tries to call pandas 'infer_freq' on the input index, which is a set of
        dates/times. If the output from the pandas function is None, then it uses a custom calculation
        to try to obtain an output that is not None.
        
        Arguments:
            allow_missing: if False, then try using the pandas 'infer_freq' method first. If False, then
            try to supplement the pandas calculation with a custom version that works if the pandas 
            infer_freq fails due to missing data (and hence would return None)
    """
    if len(list(index)) > 2:
        freq = pd.infer_freq(index)
    else:
        freq = None

    if freq is None and allow_missing:
        freq = calc_freq(index)
    return freq

def calc_freq(index):
    """ Calculate the frequency of a set of observation dates/times.
    
        Based on the spacing between the observations, estimate the frequency.
        There can be missing dates/times, and this function thus operates 
            on the min, max, and median spacings.
    """
    if isinstance(index, list):
        index = np.array(index)

    diffs = (index[1:] - index[:-1]) / pd.Timedelta(1, 'S')
    _min, _max, _med, _avg = np.min(diffs), np.max(diffs), np.median(diffs), np.mean(diffs)

    if np.isclose(_min, 1):
        freq = 'S'
    elif np.isclose(_med, 30, atol=1e-2):
        freq = '30S'
    elif np.isclose(_med, 60, atol=1e-2):
        freq = 'T'
    elif np.isclose(_med, 3600, atol=1):
        freq = 'H'
    elif np.isclose(_med, SECONDS_PER_DAY, atol=1):
        if _avg <= 1.2 * SECONDS_PER_DAY:
            freq = 'D'
        elif _avg <= 1.6 * SECONDS_PER_DAY:
            freq = 'B'
        else:
            freq = None
    elif SECONDS_PER_DAY * 5 <= _min and _med <= SECONDS_PER_DAY * 12:
        freq = 'W'
    elif SECONDS_PER_DAY * 24 <= _min  and _med <= SECONDS_PER_DAY * 35:
        freq = 'M'
    elif SECONDS_PER_DAY * 83 <= _min  and _med <= SECONDS_PER_DAY * 97:
        freq = 'Q'
    elif SECONDS_PER_DAY * 355 <= _min  and _med <= SECONDS_PER_DAY * 370:
        freq = 'Y'
    else:
        freq = None
    return freq

def is_time_series_mixed_monthly(ts):
    # If the current frequency is unknown (None), see if it is because
    #     some columns have data on the last day of the month, and other
    #     columns have data on the last BUSINESS day of the month.
    frequencies, groups = _get_mixed_monthly_frequency_groups(ts)
    return np.all([f in ('B', 'BM') for f in frequencies])

def convert_mixed_monthly_time_series(ts, target_freq):
    resampled_ts = pd.DataFrame()
    frequencies, groups = _get_mixed_monthly_frequency_groups(ts)        
    for cols in groups.values():
        sub_ts = self._load_time_series_raw(cols, frequency=target_freq, start=start, end=end)
        resampled_ts = resampled_ts.merge(sub_ts, left_index=True, right_index=True, how='outer')
    return resampled_ts

def _get_mixed_monthly_frequency_groups(ts):
    frequencies = []
    groups = defaultdict(list)
    for col in ts.columns:
        idx = ts[col].dropna().index
        frequencies.append(pd.infer_freq(idx))

        hash_key = hash(';'.join(idx.astype(str)))
        groups[hash_key].append(col)
    return frequencies, groups

def get_years_between_dates(start, end):
    if isinstance(start, (str, datetime.datetime, datetime.date)):
        start = pd.Timestamp(start)

    if isinstance(start, Iterable):
        start = pd.DatetimeIndex(start)
        
    if isinstance(end, (str, datetime.datetime, datetime.date)):
        end = pd.Timestamp(end)

    if isinstance(end, Iterable):
        end = pd.DatetimeIndex(end)
        
    res = (end - start).days / DAYS_PER_YEAR
    if isinstance(res, Iterable):
        return np.array(res, dtype=float)
    else:
        return res

def get_tenors_from_maturity_bucket(bkt):
    """ Parse a maturity bucket string (e.g., "1-10y"), and return the 
            lower and upper tenors contained in the bucket. 
        Example:
            For bucket "1-10y", this function would return the tuple ("1Y", "10Y")
        """
    res = bkt.upper().split('-')
    
    if len(res) == 2:
        low, high = res
        freq = high[-1]
        high = high[:-1]
    elif len(res) == 1:
        freq = bkt[-1].upper()
        low = high = bkt[:-1]
    else:
        raise ValueError(f'Bucket should have one or two components, separated by a hyphen "-". Provided bucket was: {bkt}')

    assert freq in ('B', 'D", ''W', 'Q', 'M', 'Y'), f'Unsupported frequency for maturity bucket: {freq}'

    low_tenor = f'{low}{freq}'
    high_tenor = f'{high}{freq}'
    return low_tenor, high_tenor

def extract_window_size(window, ts, allow_missing=True):
    """ Parse a 'window' argument to the pandas rolling/expanding fuctions.
        Allow the user to provide either an integer value (e.g. 120), or
        a string value (e.g. "10y")
        """
    if isinstance(window, (int, float, np.float32)):
        return int(np.round(window))
    elif isinstance(window, str):
        n_years_in_window = get_years_per_period(window)
        freq = infer_freq(ts.index, allow_missing=allow_missing)
        n_periods_per_year = get_periods_per_year(freq)
        return int(np.round(n_years_in_window * n_periods_per_year))
    else:
        raise ValueError(f'Unsupported data type for window size: {window}')

def get_end_of_month_date(input_dates):
    if isinstance(input_dates, str) or not isinstance(input_dates, Iterable):
        dates = [pd.Timestamp(input_dates)]
    else:
        dates = list(input_dates)

    month_ends = []
    for d in dates:
        month_end = pd.Timestamp(datetime.date(d.year, d.month, calendar.monthrange(d.year, d.month)[-1]))
        month_ends.append(month_end)
        
    if isinstance(input_dates, str) or not isinstance(input_dates, Iterable):
        return month_ends[0]
    else:
        return month_ends
