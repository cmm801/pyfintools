from abc import ABC, abstractmethod
import coinbasepro as cbp
import csv
import datetime
import httplib2
import io
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import requests_cache
import time
import urllib.request

import ibk.connect
import ibk.constants
import ibk.master


def download_time_series(data_source, base_ticker, start=None, end=None, 
                         frequency=None, period=None, session=None, max_wait_time=None):
    """ Download data for a single base ticker (which may yield one or more time series).

        Returns pandas DataFrame object with datetimes as the index and tickers as columns.
    """
    if data_source == 'gsw':
        # Daily US Government Yields from GSW/Fed
        dobj = GSWDownloader()
    elif data_source == 'fred':
        # Download all FRED tickers that are contained in our meta database
        dobj = FREDDownloader()
    elif data_source == 'acm':
        # Daily US Term Premium estimate from ACM/Fed
        dobj = ACMDownloader()
    elif data_source == 'famafrench':
        # Daily Fama French factor data
        dobj = FamaFrenchDownloader()
    elif data_source == 'yahoo':
        # Daily Yahoo data
        dobj = YahooDownloader()
    elif data_source == 'cbp':
        # Daily CoinbasePro data
        dobj = CoinbaseProDownloader()
    elif data_source == 'ib':
        # Daily CoinbasePro data
        dobj = InteractiveBrokersDownloader()
    elif data_source == 'cboe':
        # VIX Futures data
        dobj = CBOEDownloader()
    else:
        raise ValueError(f'Unknown data source: "{data_source}"')

    # Download the time series data
    ts = dobj.download_time_series(base_ticker, start=start, end=end, frequency=frequency,
                       period=period, session=session, max_wait_time=max_wait_time)
    
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise ValueError('The index of the output must be a pandas DatetimeIndex.')

    # Don't specify a name for the index values
    ts.index.name = None
    return ts


class AbstractDownloader(ABC):
    def __init__(self):
        pass

    def download_time_series(self, base_ticker, start=None, end=None, frequency=None, 
                                     period=None, session=None, max_wait_time=None):
        """ Download time series from various underlying methods.
        """
        # parse arguments and make sure they are in a consistent format
        start, end, frequency, period = self._standardize_arguments(start=start,
                                    end=end, frequency=frequency, period=period)
        
        # Get the downloaded raw time series
        raw_ts = self._download_raw_time_series(base_ticker, start=start, end=end, 
                                        frequency=frequency, period=period,
                                        session=session, max_wait_time=max_wait_time)

        # Rename any columns if necessary
        ts = self._rename_time_series(raw_ts, base_ticker)
        
        # Add any additional constructed time series
        ts_full = self._add_constructed_time_series(ts, base_ticker)
        
        # Drop any rows where all data is missing
        ts_full.dropna(inplace=True, how='all', axis=0)
        
        # Return the pandas DataFrame
        return ts_full

    @abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None, frequency=None, 
                                  period=None, session=None, max_wait_time=None):
        """ This method gets the raw time series from the data source.
        """
        raise NotImplementedError('Must be implemented by the subclass.')

    def _standardize_arguments(self, start=None, end=None, frequency=None, period=None):
        """ Standardize input arguments so the lower-level functions know what data type to expect.
        """
        if end is None:
            end = datetime.datetime.now()
            
        if frequency is None:
            frequency = '1d'

        return start, end, frequency, period

    def _add_constructed_time_series(self, ts, base_ticker):
        """ Add additional constructed time series to the ones that are downloaded.
        
            This function can be overloaded to provide construction methodologies for new time series.
            By default, no new time series are constructed and only the downloaded data is returned.
        """
        return ts

    def _rename_time_series(self, ts, base_ticker):
        """ Rename columns to use our internal names rather than those from the data source.
        """
        return ts


class PandasDatareaderDownloader(AbstractDownloader):
    @property
    def data_source(self):
        raise NotImplementedError('Must be implemented by the base class.')

    # Implement abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None, frequency=None, 
                                  period=None, session=None, max_wait_time=None):
        """ This method gets the raw time series from the data source.
        """
        if period is not None:
            raise NotImplementedError(f'"period" has value {period} but is not implemented.')

        if period is not None:
            raise NotImplementedError(f'"frequency" has value {frequency} but is not implemented.')

        data = web.DataReader(base_ticker, self.data_source, start=start, end=end, session=session)        
        return data

    def _format_input_arguments(self, start=None, end=None, frequency=None, period=None):
        """ Make adjustments to input arguments so they can be processed by pandas_datareader.
        
            Returns tuple (start, end, frequency, period)
        """
        # parse arguments and make sure they are in a consistent format
        start, end, frequency, period = super()._standardize_arguments(start=start,
                                        end=end, frequency=frequency, period=period)
        return start, end, frequency, period


class FREDDownloader(PandasDatareaderDownloader):
    @property
    def data_source(self):
        return 'fred'


class FamaFrenchDownloader(PandasDatareaderDownloader):
    COLUMN_MAP = {
        'Developed_5_Factors_daily' : {
            'MKT-RF' : 'FFDEVMKT', 'SMB' : 'FFDEVSMB', 'HML' : 'FFDEVHML',
            'RMW' : 'FFDEVRMW', 'CMA' : 'FFDEVCMA', 'RF' : 'FFDEVRF'
        },
        'Developed_Mom_Factor_daily' : {
            'WML' : 'FFDEVMOM', 'MOM' : 'FFDEVMOM'
        },
        'F-F_Research_Data_5_Factors_2x3_daily' : {
            'MKT-RF' : 'FFUSMKT', 'SMB' : 'FFUSSMB', 'HML' : 'FFUSHML',
            'RMW' : 'FFUSRMW', 'CMA' : 'FFUSCMA', 'RF' : 'FFUSRF'
        },
        'F-F_Momentum_Factor_Daily' : {
            'WML' : 'FFUSMOM', 'MOM' : 'FFUSMOM'
        },
    }

    @property
    def data_source(self):
        return 'famafrench'

    # Implement abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None, frequency=None,
                                  period=None, session=None, max_wait_time=None):
        """ This method gets the raw time series from the data source.
        """
        raw_data = super()._download_raw_time_series(base_ticker, self.data_source, 
                        start=start, end=end, freq=frequency, session=session)
            
        # Get the monthly data/daily data
        # The 0-th index is the daily or monthly data - the 1-st index is annual data
        ts = raw_data[0]

        # Reset the index to use pandas DatetimeIndex objects
        if not isinstance(ts.index, pd.DatetimeIndex):
            ts.index = ts.index.to_timestamp()
        return ts

    def _rename_time_series(self, ts, base_ticker):
        """ Rename columns to use our internal names rather than those from the data source.
        """
        ts = super()._rename_time_series(ts, base_ticker)
        
        # Get rid of whitespace in the names
        ff_ts = ts.copy()
        ff_ts.columns = [col.upper().replace(' ', '') for col in ts.columns]
        
        # Rename the columns
        ff_ts = ff_ts.rename(self.COLUMN_MAP[base_ticker], axis=1)
        return ff_ts


class YahooDownloader(PandasDatareaderDownloader):
    @property
    def data_source(self):
        return 'yahoo'

    # Overload superclass method
    def _standardize_arguments(self, start=None, end=None, frequency=None, period=None):
        """ Standardize input arguments so the lower-level functions know what data type to expect.
        """
        # Call superclass method to parse arguments and make sure they are in a standardized format.
        start, end, frequency, period = super()._standardize_arguments(start=start,
                                        end=end, frequency=frequency, period=period)
        if frequency is None:
            frequency = '1d'
        return start, end, frequency, period

    def _rename_time_series(self, ts, base_ticker):
        """ Rename columns to use our internal names rather than those from the data source.
        """
        ts = super()._rename_time_series(ts, base_ticker)

        # Rename the columns
        col_map = {'Open' : f'{base_ticker}(PO)',
                   'Close' : f'{base_ticker}(PC)',
                   'High' : f'{base_ticker}(PH)',
                   'Low' : f'{base_ticker}(PL)',
                   'Volume' : f'{base_ticker}(VO)',
                   'Adj Close' : f'{base_ticker}(RI)',
                  }        
        ts = ts.rename(col_map, axis=1)
        return ts


class GSWDownloader(AbstractDownloader):
    URL_GSW = 'https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv'
    
    # Implement abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None, frequency=None,
                                  period=None, session=None, max_wait_time=None):
        """ This method gets the raw time series for the GSW US yield curve data.
        
            The base ticker gets ignored in this class, since there is only 1 dataset.
        """
        header_row = self._find_gsw_header_row()
        ts = pd.read_csv(self.URL_GSW, skiprows=header_row, index_col=0)
        
        # Make sure the index is a pandas DatetimeIndex
        ts.index = pd.DatetimeIndex(ts.index)
        return ts

    def _rename_time_series(self, ts, base_ticker):
        """ Rename columns to use our internal names rather than those from the data source.
        """
        ts = super()._rename_time_series(ts, base_ticker)
        ts.columns = 'GSW' + ts.columns
        return ts

    def _find_gsw_header_row(self):
        """ Find the header row for the .csv file containing the GSW yields.
        """
        webpage = urllib.request.urlopen(self.URL_GSW)
        datareader = csv.reader(io.TextIOWrapper(webpage))

        header_row = 0
        while header_row < 11 and datareader:
            txt = next(datareader)
            if txt and txt[0].lower() == 'date':
                break
            else:
                header_row += 1
        return header_row


class ACMDownloader(AbstractDownloader):
    ACM_URL = 'https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls'
    ACM_SHEETNAME = 'ACM Daily'
    ACM_INDEX_COL = 'DATE'
    
    # Implement abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None, frequency=None,
                                  period=None, session=None, max_wait_time=None):
        """ This method gets the raw time series for the ACM term premium data.
        
            The base ticker gets ignored in this class, since there is only 1 dataset.        
        """ 
        ts = pd.read_excel(self.ACM_URL, sheet_name=self.ACM_SHEETNAME)
        ts = ts.set_index(self.ACM_INDEX_COL)
        ts.index = pd.DatetimeIndex(ts.index)
        return ts


class CoinbaseProDownloader(AbstractDownloader):
    ALLOWED_GRANULARITY = (60, 300, 900, 3600, 21600, 86400,)
    MAX_POINTS_PER_REQUEST = 300
    MAX_REQUEST_PER_SECOND = 3
    DEFAULT_START_DATE = datetime.datetime(2015, 12, 31) 

    # Implement abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None, frequency=None,
                                  period=None, session=None, max_wait_time=None):
        """ Download CoinbasePro time series for a single base ticker.
        
            Arguments:
                frequency: must be one of '1min', '5min', '15min', '1h', '6h', '1d'
        """
        client = cbp.PublicClient()
        
        # Get the 'granularity', which is number of seconds in one period with the given frequency
        granularity = int(pd.Timedelta(frequency).total_seconds())
        if granularity not in self.ALLOWED_GRANULARITY:
            raise ValueError(f'Granularity {granularity} is not in ' +\
                             f'range of allowed values {self.ALLOWED_GRANULARITY}')
        
        period_start = start
        period_end = min(period_start + datetime.timedelta(seconds=granularity * self.MAX_POINTS_PER_REQUEST), end)
        ts_list = []
        while period_start < end:
            # Make sure to space out the requests
            request_spacing = datetime.timedelta(seconds=1/self.MAX_REQUEST_PER_SECOND)

            success = False
            while not success:
                try:
                    # Try to get raw data from the API
                    raw_data = client.get_product_historic_rates(base_ticker, start=period_start, 
                                                stop=period_end, granularity=granularity)
                except cbp.exceptions.RateLimitError:
                    time.sleep(1/self.MAX_REQUEST_PER_SECOND)
                except KeyboardInterrupt:
                    # Allow the user to exit via the keyboard interrupt...
                    raise KeyboardInterrupt
                except:
                    print('handling error')
                else:
                    success = True

            # Convert the data into a pandas DataFrame object
            sub_ts = pd.DataFrame(raw_data)
            ts_list.append(sub_ts)

            # Update the start/end times
            period_start = period_end + datetime.timedelta(seconds=granularity)
            period_end  = min(period_start + datetime.timedelta(seconds=granularity * self.MAX_POINTS_PER_REQUEST), end)

        # Combine the time series into a single object
        ts = pd.concat(ts_list, axis=0)
        
        # Define the list of expected columns
        expected_columns = set(['high', 'low', 'open', 'close', 'volume', 'time'])

        # Set the index
        if ts.shape[0]:
            if set(ts.columns) != expected_columns:
                raise ValueError('Unexpected or missing columns from CoinbasePro.')
            else:
                ts = ts.set_index('time')
        else:
            ts = pd.DataFrame([], columns=expected_columns).set_index('time')
        
        # Convert the data to a data frame and convert the columns to float
        for col in ts.columns:
            ts[col] = ts[col].astype('float')

        # Sort the dates in ascending order
        ts = ts.sort_index()
        return ts

    # Overload superclass method
    def _standardize_arguments(self, start=None, end=None, frequency=None, period=None):
        """ Standardize input arguments so the lower-level functions know what data type to expect.
        
            Make sure start/end are datetime objects.
        """
        # Call superclass method to parse arguments and make sure they are in a standardized format.
        start, end, frequency, period = super()._standardize_arguments(start=start,
                                        end=end, frequency=frequency, period=period)

        # Make sure the 'start' argument is a datetime object
        if start is None:
            ST = self.DEFAULT_START_DATE
        elif not isinstance(start, (datetime.datetime, datetime.date)):
            ST = datetime.datetime.fromisoformat(start)
        else:
            ST = start

        # Make sure the 'end' argument is a datetime object
        if end is None:
            ET = datetime.datetime.now()
        elif not isinstance(end, (datetime.datetime, datetime.date)):
            ET = datetime.datetime.fromisoformat(end)
        else:
            ET = end

        # Make sure the frequency is not None
        if frequency is None:
            frequency = '1d'
            
        # Check if period is provided
        if period is not None:
            raise NotImplementedError(f'"period" has value {period} but logic is not implemented.')
            
        return ST, ET, frequency, period

    def _rename_time_series(self, ts, base_ticker):
        """ Rename columns to use our internal names rather than those from the data source.
        """
        # Call the superclass method
        ts = super()._rename_time_series(ts, base_ticker)

        # Get the map from the CBP names to our internal names
        series_type_map = pd.Series({'open' : 'PO', 'close' : 'PC', 
                                     'high' : 'PH', 'low' : 'PL', 'volume' : 'VO'})

        # Rename the columns
        series_types = series_type_map[ts.columns]
        tickers = [f'{base_ticker}({series_type})' for series_type in series_types]
        ts.columns = tickers
        return ts


class InteractiveBrokersDownloader(AbstractDownloader):
    _app = None

    @property
    def app(self):
        if self._app is not None:
            return self._app
        else:
            # First try connecting to the paper portfolio
            self._app = ibk.master.Master(port=ibk.constants.PORT_PAPER)

            # Check if we can access market data from the paper portfolio
            try:
                self._app.marketdata_app
                
                # If this worked, then we can return the class handle
                return self._app
            except ibk.connect.ConnectionNotEstablishedError:
                self._app = ibk.master.Master(port=ibk.constants.PORT_PROD)
                
                # Try to access market data from the production portfolio
                try:
                    self._app.marketdata_app
                    
                    # If this worked, then we can return the class handle
                    return self._app                    
                except ibk.connect.ConnectionNotEstablishedError:
                    raise ValueError('No connection to Interactive Brokers is detected. Try logging in.')

    # Implement abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None, frequency=None, 
                                  period=None, session=None, max_wait_time=None):
        """ This method downloads the raw time series using the InteractiveBroker's API.
        
            Arguments:
                base_ticker: (str) the IB ticker (localSymbol) associated with the time series.
                    For continuous futures contracts, us the ticker followed by _CNT. For example,
                    'ES_CNT' would the the base ticker to get continuous futures data for 'ES'.
                start/end: the start/end date/time for the historical request
                frequency: (str) the frequency of the data to be returned
        """ 
        # Get the relevant contract for the ticker symbol
        if base_ticker.endswith('_CNT'):
            # If the ticker ends with _CNT, then the user is requesting a continuous futures contract
            underlier_ticker = base_ticker[:-4]
            contract_details = self.app.get_continuous_futures_contract_details(underlier_ticker)
            _contract = contract_details.contract
        else:
            # ...otherwise, get a normal contract
            _contract = self.app.get_contract(base_ticker)
            
        contractList = [_contract]
        
        # Create a historical request object
        reqObjList = self.app.get_historical_data(contractList, start=start, end=end,
                                             frequency=frequency, use_rth=False, data_type="TRADES")
        reqObj = reqObjList[0]
        
        # Wait until the request is complete
        if max_wait_time is None:
            max_wait_time = 1e6   # Wait very long time if no max. wait time is specified
        t0 = time.time()
        while not reqObj.is_request_complete() and time.time() - t0 < max_wait_time:
            time.sleep(0.1)
            
        return reqObj.get_dataframe(timestamp=False)

    # Overload superclass method
    def _rename_time_series(self, ts, base_ticker):
        # Get the map from the CBP names to our internal names
        series_type_map = pd.Series({'open' : 'PO', 'close' : 'PC', 
                                     'high' : 'PH', 'low' : 'PL', 'volume' : 'VO',
                                     'barCount' : 'BARCT', 'average' : 'VWAP'})

        series_types = series_type_map[ts.columns.values]
        tickers = [f'{base_ticker}({series_type})' for series_type in series_types]
        ts.columns = tickers
        return ts
    
    # Overload superclass method
    def _standardize_arguments(self, start=None, end=None, frequency=None, period=None):
        # Call the superclass method first
        start, end, frequency, period = super()._standardize_arguments(start=start, end=end,
                                                                       frequency=frequency, period=period)
        
        # Convert the arguments into the format the IB requires
        ST = start.strftime('%Y%m%d %H:%M:%S')
        ET = end.strftime('%Y%m%d %H:%M:%S')        
        return ST, ET, frequency, period


class CBOEDownloader(AbstractDownloader):
    BASE_URL = 'https://markets.cboe.com/us/futures/market_statistics/historical_data/products/csv/VX'

    # Implement abstractmethod
    def _download_raw_time_series(self, base_ticker, start=None, end=None, frequency=None, 
                                  period=None, session=None, max_wait_time=None):
        """ This method downloads time series from .csv files on the CBOE website.
        
            Arguments:
                base_ticker: (str) a string containing the futures
                    ticker symbol (e.g. 'VX' or 'ES') and the expiry datetime of the contract.
                    For example, this would be VX20210317 for a VIX future expiring 2021-03-17.
                start/end: the start/end date/time for the historical request
                frequency: (str) the frequency of the data to be returned. Default is daily data.
        """ 
        # Parse the symbol to extract the expiration date
        symbol, expiry = self._parse_ticker(base_ticker)

        if expiry is None:
            df = self._download_index_time_series(symbol)
        else:
            df = self._download_futures_time_series(symbol, expiry)
            
        if start is not None:
            df = df.loc[pd.Timestamp(start) <= df.index]

        if end is not None:
            df = df.loc[df.index <= pd.Timestamp(end)]

        return df
    
    def _download_index_time_series(self, symbol):
        if symbol != 'VIX':
            raise NotImplementedError('Only VIX futures data is currently available.')

        filename_curr = 'https://ww2.cboe.com/publish/scheduledtask/mktdata/datahouse/vixcurrent.csv'
        df = pd.read_csv(filename_curr, header=1)

        df.set_index('Date', inplace=True)
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = 'pricing_date'
        return df

    def _download_futures_time_series(self, symbol, expiry): 
        """ Download futures time series. """
        if symbol != 'VX':
            raise NotImplementedError('Only VIX futures data is currently available.')
            
        # Check if the URL exists for this expiry date
        url = f'{self.BASE_URL}/{expiry}/'
        http = httplib2.Http()
        resp = http.request(url, 'HEAD')
        if int(resp[0]['status']) >= 400:
            raise ValueError(f'No futures data found for "{symbol}" expiry: "{expiry}"')
        else:
            df = pd.read_csv(url)
            df.drop(['Change', 'Futures'], axis=1, inplace=True)

        # Set the index
        df.set_index('Trade Date', inplace=True)
        df.index = pd.DatetimeIndex(df.index)
        df.index.name = 'pricing_date'
        return df

    # Overload superclass method
    def _rename_time_series(self, ts, base_ticker):
        # Get the map from the CBP names to our internal names
        series_type_map = pd.Series({'Open' : 'PO', 'VIX Open' : 'PO', 
                                     'Close' : 'PC', 'VIX Close' : 'PC',
                                     'High' : 'PH', 'VIX High' : 'PH',
                                     'Low' : 'PL', 'VIX Low' : 'PL',
                                     'Settle' : 'PS',
                                     'Total Volume' : 'VO',
                                     'EFP' : 'EFP', 'Open Interest' : 'OI'})

        series_types = series_type_map[ts.columns.values]
        tickers = [f'{base_ticker}({series_type})' for series_type in series_types]
        ts.columns = tickers
        return ts

    def _parse_ticker(self, base_ticker):
        """ Extract the symbol and expiry date (if there is one) from the base ticker. """
        if len(base_ticker) == 3:
            # Index data
            symbol = base_ticker
            expiry = None
        else:
            # Futures data
            symbol = base_ticker[:2]
            expiry = pd.Timestamp(base_ticker[2:]).strftime('%Y-%m-%d')

        return symbol, expiry
