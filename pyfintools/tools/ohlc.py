""" Data structures for working with OHLC data.

"""
import numpy as np
import pandas as pd

# Column names that can appear in the time series
COL_OPEN = 'open'
COL_CLOSE = 'close'
COL_HIGH = 'high'
COL_LOW = 'low'
COL_VOLUME = 'volume'
COL_OPEN_INTEREST = 'open_interest'

# All columns used by a non-derivative asset
COL_CORE = [COL_OPEN, COL_CLOSE, COL_HIGH, COL_LOW, COL_VOLUME]

# All columns used by futures/options
COL_DER = [COL_CORR, COL_OPEN_INTEREST]


class OHLC:
    def __init__(self, ts, exchange=None):
        self._timeseries = None
        self._timeseries = ts
        self.exchange = exchange
        
    @property
    def timeseries(self):
        return self._timeseries
    
    @timeseries.setter
    def timeseries(self, ts):
        self._timeseries = self._format_input_timeseries(ts)
        self.frequency = self._calc_frequency(ts)
        
    def _format_input_timeseries(self, ts):
        """ Method for putting an input time series into the standard format.
        """
        if not isinstance(ts.index, pd.DateTimeIndex):
            raise ValueError('The timeseries index must be a pandas DateTimeIndex.')

        unknown = list(set(ts.columns) - set(COL_DER))
        if unknown:
            raise ValueError('Unknown columns: {}'.format(unknown))

        missing = list(set(COL_CORE) - set(ts.columns))
        if missing:
            raise ValueError('Missing columns: {}'.format(missing))

        # Make sure the columns are ordered
        if COL_OPEN_INTEREST in ts.columns:
            ts = ts[COL_DER]
        else:
            ts = ts[COL_CORE]
            
        # Save the time series data
        self._timeseries = ts

    def _calc_frequency(self, ts):
        """ Method for obtaining the frequency of a time series.
        """
        raise NotImplementedError('Need to implement method.')

    def ffill(self):
        """ Method to fill missing data points with previous values. """
        raise NotImplementedError('Need to implement method.')
    
    
    