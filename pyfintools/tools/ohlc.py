""" Data structures for working with OHLC data.

"""
import numpy as np
import pandas as pd
import talib

import cfin
import pyfintools.tools.freq
import pyfintools.tools.plot
import pyfintools.tools.tradingutils


# Column names that can appear in the time series
COL_OPEN = 'open'
COL_CLOSE = 'close'
COL_HIGH = 'high'
COL_LOW = 'low'
COL_VOLUME = 'volume'

# Column used just for futures/options
COL_OPEN_INTEREST = 'open_interest'

# Extra columns provided by IB
COL_AVERAGE = 'average'
COL_BAR_COUNT = 'bar_count'

# All columns used by a non-derivative asset
COL_CORE = (COL_OPEN, COL_CLOSE, COL_HIGH, COL_LOW, COL_VOLUME,)

# All columns used by futures/options
COL_DER = COL_CORE + (COL_OPEN_INTEREST,)

# Optional columns that can be included
COL_OPTIONAL = (COL_AVERAGE, COL_BAR_COUNT,)

# Default value of the absolute tolerance (for prices)
DEFAULT_ABS_TOL = 1e-6


class OHLC:
    def __init__(self, ts, exchange=None, atol=None):
        self._timeseries = None
        self.timeseries = ts.copy().sort_index()
        self.exchange = exchange
        
        if atol is None:
            self.atol = DEFAULT_ABS_TOL
        else:
            self.atol = atol
        
    @property
    def timeseries(self):
        return self._timeseries
    
    @timeseries.setter
    def timeseries(self, ts):
        self._timeseries = self._format_input_timeseries(ts)
        self._timestamps = None
        self.frequency = self._calc_frequency(ts)
        
    @property
    def open(self):
        return self.timeseries.open
        
    @property
    def close(self):
        return self.timeseries.close
        
    @property
    def high(self):
        return self.timeseries.high
        
    @property
    def low(self):
        return self.timeseries.low
        
    @property
    def volume(self):
        return self.timeseries.volume
        
    @property
    def open_interest(self):
        return self.timeseries.open_interest
        
    @property
    def average(self):
        return self.timeseries.average

    @property
    def bar_count(self):
        return self.timeseries.bar_count

    @property
    def index(self):
        """ Returns index of underlying timeseries DataFrame. """
        return self.timeseries.index

    @property
    def values(self):
        """ Returns values of underlying timeseries DataFrame. """
        return self.timeseries.values

    @property
    def columns(self):
        """ Returns columns of underlying timeseries DataFrame. """
        return self.timeseries.columns

    @property
    def size(self):
        """ Returns size of underlying timeseries DataFrame. """
        return self.timeseries.size

    @property
    def shape(self):
        """ Returns shape of underlying timeseries DataFrame. """
        return self.timeseries.shape

    @property
    def timestamps(self):
        if self._timestamps is None:
            self._timestamps = np.array([d.timestamp() for d in self.index], dtype=float)
        return self._timestamps
        
    def downsample(self, frequency):
        """ Downsample the OHLC time series to the new frequency. """
        new_ts = pyfintools.tools.tradingutils.downsample(self.timeseries, frequency)
        return OHLC(new_ts, exchange=self.exchange)
    
    def calc_triple_barrier(self, lb_rtn, ub_rtn):
        """ Calculate the triple barrier for a given upside/downside target. """
        return cfin.calc_triple_barrier(close=self.close.values, 
                                        low=self.low.values, 
                                        high=self.high.values, 
                                        lb_rtn=lb_rtn, ub_rtn=ub_rtn)

    def to_volume_bars(self, bar_size, min_bar_size=0):
        """ Create a new  OHLC object with volume bars. """
        new_ts = pyfintools.tools.tradingutils.create_volume_bars(self.timeseries, bar_size, 
                                                                  min_bar_size=min_bar_size)
        return OHLC(new_ts, exchange=self.exchange)
    
    def head(self, N=None):
        """ Get an OHLC object with just the first N rows of time series data. """
        return OHLC(self.timeseries.head(N), exchange=self.exchange)

    def tail(self, N=None):
        """ Get an OHLC object with just the last N rows of time series data. """
        return OHLC(self.timeseries.tail(N), exchange=self.exchange)

    def between_time(self, start, end):
        """ Get an OHLC object with just observations between start/end time. """
        return OHLC(self.timeseries.between_time(start, end), exchange=self.exchange)

    def copy(self):
        """ Create a deep copy of the object. """
        return OHLC(self.timeseries.copy(), exchange=self.exchange)

    def ffill(self):
        """ Method to fill missing data points with previous values. """
        raise NotImplementedError('Need to implement method.')

    def plot_candlestick(self, include_volume=False):
        if not include_volume:
            pyfintools.tools.plot.plot_candlestick(self.timeseries, 
                            open_col=COL_OPEN, high_col=COL_HIGH, 
                            low_col=COL_LOW, close_col=COL_CLOSE, time_col=None)
        else:
            pyfintools.tools.plot.plot_candlestick_volume(self.timeseries, 
                            open_col=COL_OPEN, high_col=COL_HIGH, low_col=COL_LOW, 
                            close_col=COL_CLOSE, vol_col=COL_VOLUME, time_col=None)

    def OBV(self):
        """ On Balance Volume. """
        return talib.OBV(self.close, self.volume)
    
    def ATR(self, timeperiod=None):
        """ Average True Range. Default time period is 14. """
        if timeperiod is not None:
            return talib.ATR(self.high, self.low, self.close, timeperiod=timeperiod)
        else:
            return talib.ATR(self.high, self.low, self.close)

    def TRANGE(self):
        """ True Range. """
        return talib.TRANGE(self.high, self.low, self.close)

    def ADOSC(self, fastperiod=3, slowperiod=10):
        """ Chaikin A/D (Accumulation/Distribution) Oscillator.
            
            Arguments:
                fastperiod: (int) size of the fast window. Default is 3
                slowperiod: (int) size of the fast window. Default is 10
        """        
        return talib.ADOSC(high=self.high, low=self.low, close=self.close, volume=self.volume, 
                           fastperiod=fastperiod, slowperiod=slowperiod)
    
    def AD(self):
        """ Chaikin A/D (Accumulation/Distribution) line. """
        return talib.AD(high=self.high, low=self.low, close=self.close, volume=self.volume)

    def _format_input_timeseries(self, ts):
        """ Method for putting an input time series into the standard format.
        """
        if not isinstance(ts.index, pd.DatetimeIndex):
            raise ValueError('The timeseries index must be a pandas DatetimeIndex.')

        unknown = set(ts.columns) - set(COL_DER + COL_OPTIONAL)
        if unknown:
            raise ValueError('Unknown columns: {}'.format(unknown))

        missing = set(COL_CORE) - set(ts.columns)
        if missing:
            raise ValueError('Missing columns: {}'.format(missing))

        # Save the time series data
        return ts

    def _calc_frequency(self, ts):
        """ Method for obtaining the frequency of a time series.
        """
        return pyfintools.tools.freq.infer_freq(ts.index, allow_missing=True)
