import os
import time
import datetime
import pytz
import re

import pandas as pd
import numpy as np

import ibk.constants
import ibk.helper


def _get_valid_files(ticker, frequency, start, end, data_type):
    """Retrieve all .csv files within a time period for a given ticker.
    """
    all_files = sorted(os.listdir(IB_DATA_PATH))
    valid_files = []
    for f in all_files:
        parsed = re.split('[\._]', f)
        if parsed[-1] == 'csv' and [parsed[j] for j in [0, 1, 3]] == [ticker, frequency, data_type]:
            dt = datetime.datetime.strptime(parsed[2], '%Y%m%d')
            if start <= dt <= end:
                valid_files.append(IB_DATA_PATH + f)

    return valid_files

def _convert_timestamp_to_date(d):
    est_date = ibk.helper.convert_utc_timestamp_to_datetime(d, tz_name=ibk.constants.TIMEZONE_EST)
    return ibk.helper.convert_datetime_to_tws_date(est_date)

def convert_tws_date_to_utc_datetime(d):
    est_dates = ibk.helper.convert_tws_date_to_datetime(d, tz_name=ibk.constants.TIMEZONE_EST)
    return est_dates.astimezone(pytz.utc)

def get_index_at_horizon(df, horizon, tol):
    """If the data are irregularly spaced in time, determine the index that 
       represents 'horizon' steps into the future.
       """
    T = df.shape[0]

    idx_R = 0
    last = None
    y = np.nan * np.zeros((T,), dtype=np.float32)
    timestamps = df.index.values
    for idx_L in range(T):
        while idx_R < T and timestamps[idx_R] - timestamps[idx_L] < horizon:
            idx_R += 1

        if idx_R >= T:
            break
        elif timestamps[idx_R] - timestamps[idx_L] == horizon:
            y[idx_L] = idx_R
        else:
            diff_plus = timestamps[idx_R] - timestamps[idx_L]
            diff_minus = timestamps[idx_R-1] - timestamps[idx_L]
            if diff_plus <= diff_minus and abs(diff_plus - horizon) <= tol:
                y[idx_L] = idx_R
            elif diff_minus <= diff_plus and abs(diff_minus - horizon)  <= tol:
                y[idx_L] = idx_R - 1
    return y

def save_daily_ib_data(df, ticker, frequency, tws_date, data_type):
    """Save a DataFrame of IB historical daily data in the proper format."""
    filename = 'data/{}_{}_{}_{}.csv'.format(ticker, frequency, tws_date, data_type)
    assert df.index.name == 'utc_timestamp', \
                    'The data frame must include the UTC timestamp as an index to be saved.'
    df.to_csv(filename, index=True)
    return filename

def get_historical_ib_data(ticker, frequency, start, end, data_type, fill_missing=True):
    # Create a DataFrame with all relevant data 
    files = _get_valid_files(ticker=ticker, frequency=frequency, start=start, end=end, data_type=data_type)
    df = pd.DataFrame()
    for f in files:
        tmp_df = pd.read_csv(f)
        df = pd.concat([df, tmp_df], sort=True)

    df.set_index('utc_timestamp', inplace=True)
    df.sort_index(inplace=True)     
    df.drop_duplicates(inplace=True)

    # Remove extra whitespace in the date
    df.date = [re.sub('\s+',' ', d) for d in df.date.values]
    
    if fill_missing:
        # Add rows for all missing observations
        idx_start = df.index.values[0]
        idx_end = df.index.values[-1]
        all_indices = set(np.arange(idx_start, idx_end, 1.0))
        missing_indices = all_indices - set(df.index.values)
        empty_data = np.nan * np.zeros((len(missing_indices), df.shape[1]), dtype=np.float32)
        empty_df_index = pd.Index(missing_indices, name=df.index.name)
        empty_df = pd.DataFrame(empty_data, index=empempty_df_index, columns=df.columns)
        empty_df.volume = 0
        empty_df.barCount = 0
        empty_df.date = [_convert_timestamp_to_date(t) for t in empty_df.index.values]

        # Add the missing timestamps to the Data Frame
        df = pd.concat([empty_df, df]).sort_index()
        df.close.ffill(inplace=True)

        # Replace missing price columns with the previous close when no trades have occurred
        idx_missing = np.isnan(df.low.values)
        df.loc[idx_missing, 'average'] = df.loc[idx_missing, 'close'].values
        df.loc[idx_missing, 'open'] = df.loc[idx_missing, 'close'].values
        df.loc[idx_missing, 'high'] = df.loc[idx_missing, 'close'].values
        df.loc[idx_missing, 'low'] = df.loc[idx_missing, 'close'].values
        
    return df

def create_volume_bars(input_df, bar_size, min_bar_size=0):
    """Create volume bars from a set of time bars.
       NOTE: Following the TWS convention, the timestamp and date refer to the BEGINNING of each bar, not the end.
       """    
    # We need to add a single repeated row at the beginning of the DataFrame
    #   which will make the algorithm more simple
    new_row = input_df.iloc[[0]].copy()
    new_row.loc[:,'volume'] = 0
    new_row.loc[:,'barCount'] = 0
    new_date = pd.DatetimeIndex(input_df.iloc[[0]].date).to_pydatetime()[0] - datetime.timedelta(seconds=1)
    new_row.loc[:,'date'] = new_date
    new_row.index.values[0] = input_df.index.values[0] - 1

    # Append new row to the beginning of the data frame
    df = pd.concat([new_row, input_df])
    T = df.shape[0]
    t_start = 1

    vol_times_prc = df.volume.values * df.average.values
    expanding_vol = df.volume.expanding().sum()
    vol_bars = []
    for t in range(1, T+1):
        t_prev = t_start - 1
        bar_vol_prev = expanding_vol.values[t-1] - expanding_vol.values[t_prev]
        bar_vol_curr = bar_vol_prev + df.volume.values[t] if t < T else float('inf')
        if (bar_vol_prev > min_bar_size and bar_vol_curr - bar_size > abs(bar_vol_prev - bar_size)) \
                    or (t == T and bar_vol_prev > 0):
            # Adding the next tick would make the bar too big, so we save the bar
            vol_bars.append(dict(utc_timestamp=df.index.values[t_start], 
                                 volume=bar_vol_prev,
                                 average=vol_times_prc[t_start:t].sum() / bar_vol_prev,
                                 barCount=df.barCount.values[t_start:t].sum(),
                                 date=df.date.values[t_start],
                                 open=df.open.values[t_start],
                                 high=df.high.values[t_start:t].max(),
                                 low=df.low.values[t_start:t].min(),
                                 close=df.close.values[t-1]
                                ))
            t_start = t
            vol = 0

    output = pd.DataFrame(vol_bars)
    output.set_index('utc_timestamp', inplace=True)
    return output

def convert_tick_to_time_bars(input_df):
    """ Downsample a series of ticks into a DataFrame of 1s bars. 
    """
    def _open_fun(x): return x.values[0]
    def _close_fun(x): return x.values[-1]
    def _count_fun(x): return x.shape[0]

    input_df['vol_price'] = input_df['price'] * input_df['size']
    agg_rules = dict(
                     price=[_open_fun, np.max, np.min, _close_fun],
                     size=[np.sum, _count_fun],
                     vol_price=np.sum
                    )

    ts = input_df.groupby(input_df.index.values).agg(agg_rules)
    ts.columns = ['open', 'high', 'low', 'close', 'volume', 'barCount', 'vol_price']
    ts['average'] = ts.vol_price / ts.volume
    
    # Drop the extra column (volume * price) that was added for the calculation of VWAP
    ts.drop('vol_price', axis=1, inplace=True)
    input_df.drop('vol_price', axis=1, inplace=True)
    return ts
    
def downsample(input_df, frequency):
    """ Downsample a data frame from a higher to a lower frequency.
    """
    col_names = input_df.columns
    seconds_per_bar = ibk.helper.TimeHelper(frequency, 'frequency').total_seconds()
    if 'volume' in col_names:
        input_df = input_df.iloc[input_df.volume.values > 0]
        
    # We need to add a single repeated row at the beginning of the DataFrame
    #   which will make the algorithm more simple
    new_row = input_df.iloc[[0]].copy()
    new_row.index.values[0] = input_df.index.values[0] - 1

    # Append new row to the beginning of the data frame
    df = pd.concat([new_row, input_df])

    if 'volume' in col_names:
        df['volume'].iloc[0] = 0
        expanding_vol = df.volume.expanding().sum()
        vol_times_prc = df.volume.values * df.average.values

    T = df.shape[0]
    t_start = t_end = 1
    bar_start_time = int(input_df.index.values[0] / seconds_per_bar) * seconds_per_bar

    time_bars = []
    while bar_start_time < df.index.values[-1]:
        bar_end_time = min(bar_start_time + seconds_per_bar, df.index.values[-1] + 1)        
        while t_end < T and df.index.values[t_end] < bar_end_time:
            t_end += 1

        # Adding the next tick would make the bar too big, so we save the bar
        bar = dict(utc_timestamp=bar_start_time,
                   open=df.open.values[t_start],
                   high=df.high.values[t_start:t_end].max(),
                   low=df.low.values[t_start:t_end].min(),
                   close=df.close.values[t_end-1])

        if 'barCount' in col_names:
            bar['barCount'] = df.barCount.values[t_start:t_end].sum()
        if 'volume' in col_names:
            bar['volume'] = int(expanding_vol.values[t_end - 1] - expanding_vol.values[t_start - 1])
        if 'average' in col_names:
            bar['average'] = vol_times_prc[t_start:t_end].sum() / bar['volume']

        time_bars.append(bar)
        t_start = t_end  
        if t_start == T:
            bar_start_time = df.index.values[-1]
        else:
            while bar_start_time + seconds_per_bar <= df.index.values[t_start]:
                bar_start_time += seconds_per_bar


    # Create a DataFrame for output and set the index to be the UTC timestamp
    output = pd.DataFrame(time_bars)
    output.set_index('utc_timestamp', inplace=True)    
    tz_est = ibk.constants.TIMEZONE_EST    

    # Get the new dates based on the timestamp
    dates = []
    for idx in output.index.values:
        dt = ibk.helper.convert_utc_timestamp_to_datetime(idx, tz_name=tz_est)
        dates.append(ibk.helper.convert_datetime_to_tws_date(dt))
    output['date'] = dates

    # Put the columns back into the original order before returning
    output = output.loc[:,input_df.columns]
    return output

def set_timestamp_index(df):
    """ Set the index of the data frame to be the UTC timestamp.
    """
    if 'utc_timestamp' != df.index.name:
        if 'utc_timestamp' not in df.columns:
            utc_datetimes = [convert_tws_date_to_utc_datetime(d) for d in df.date.values]
            utc_timestamps = [dt.timestamp() for dt in utc_datetimes]
            df.index = pd.Index(utc_timestamps, name='utc_timestamp')
        else:
            df.set_index('utc_timestamp', inplace=True) 
            
def add_missing_rows(_df, t_start, t_end):
    """Add missing rows between t_start and t_end to a data frame.
    """
    ind_vals = np.arange(t_start, t_end + 1)
    N = ind_vals.size - _df.shape[0]
    missing_idx = set(ind_vals) - set(_df.index.values)
    empty_df = pd.DataFrame(np.nan * np.zeros((N, _df.shape[1]), dtype=np.float32), columns=_df.columns)
    empty_df['volume'] = 0
    empty_df['barCount'] = 0
    empty_df.index = pd.Index(missing_idx)
    full_df = pd.concat([_df, empty_df]).sort_index().ffill()
    return full_df