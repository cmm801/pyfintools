""" Provides unified access to time series and meta data for single securities.

    This module is intended to provide a set of classes that allow the user
    to easily switch between time series and meta data for a single financial
    security. The base class Security is then inherited by all subclasses,
    which extend its variables and functionalities depending on the specifics
    of the asset class. 
        
    Examples of single financial securities (as far as this module is concerned) are:
    - MSCI EMU index denominated in USD
    - MSCI EMU index denominated in EUR
    - MSCI EMU index hedged to USD
    - US CPI
    - Germany 10y Benchmark Bond Index
    - Canada 3m government interest rate
    
    All classes defined in this module can be initialized via the constructor by providing
    three arguments (all of which can be None if not available):
        security_info: a pandas DataFrame containing key information
        ticker_info: a pandas DataFrame containing information about which types of time
            series data are available for the given security
        ts_db: a Database object (defined in the 'database' module), that gives access
            to time series data.

    The numerous classes here inherit from the base class Security, and include additional
    variables and behaviours as necessary.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import datetime
import copy

import pyfintools.constants
import pyfintools.tools.freq
import pyfintools.security.constants
import pyfintools.security.helper


class Security(object):
    """ A class that offers unified access to time series and meta data for single securities.

        This base class is intended to be inherited by a subclasses that allow the user
        to easily switch between time series and meta data for a single financial
        security.
            
        Attributes:
            security_info: (DataFrame) contains key meta data about the security
            ticker_info: (DataFrame) constains information about which 
                    types of time series data are available for the given security
            ts_db: (Database) a Database object that provides access to time series information
            is_tradable: (bool) describes whether the Security is a tradable financial instrument 
                    (e.g. True for a single bond, but False for a bond index)
            sec_code: (str) a unique security code
            name: (str) the name of the security
            category_1_code: (str) the category 1 code of the security (e.g. EQ, BD, FX, etc)
            category_2_code: (str) the category 2 code of the security (e.g. FXSpot, EQI, etc.)
            category_3_code: (str) the category 3 code of the security
    """

    def __init__(self, security_info=None, ticker_info=None, ts_db=None):
        """ Initialize Security object.
        
            Arguments:
                security_info: a pandas DataFrame containing key information
                ticker_info: a pandas DataFrame containing information about which types of time
                    series data are available for the given security
                ts_db: a Database object (defined in the 'database' module), that gives access
                to time series data.
        """

        super().__init__()
        # Store the ticker info
        self.ticker_info = ticker_info
        if security_info is None:
            self.security_info = dict()
        else:
            self.security_info = security_info

        # Initialize the time series database information
        self._ts_db = None
        self.ts_db = ts_db

    @property
    def is_tradable(self):
        return self._get_property_value('is_tradable', default_val=None)

    @is_tradable.setter
    def is_tradable(self, val):
        assert isinstance(val, bool), 'is_tradable must be True/False.'
        self.security_info['is_tradable'] = val

    @property
    def sec_code(self):
        return self._get_property_value('sec_code', default_val='')

    @sec_code.setter
    def sec_code(self, val):
        assert isinstance(val, str), 'sec_code must be a string.'
        self.security_info['sec_code'] = val

    @property
    def name(self):
        return self._get_property_value('name', default_val='')

    @name.setter
    def name(self, val):
        assert isinstance(val, str), 'name must be a string.'
        self.security_info['name'] = val

    @property
    def category_1_code(self):
        return self._get_property_value('category_1_code', default_val='')

    @category_1_code.setter
    def category_1_code(self, val):
        assert isinstance(val, str), 'category_1_code must be a string.'
        self.security_info['category_1_code'] = val

    @property
    def category_2_code(self):
        return self._get_property_value('category_2_code', default_val='')

    @category_2_code.setter
    def category_2_code(self, val):
        assert isinstance(val, str), 'category_2_code must be a string.'
        self.security_info['category_2_code'] = val

    @property
    def category_3_code(self):
        return self._get_property_value('category_3_code', default_val='')

    @category_3_code.setter
    def category_3_code(self, val):
        assert isinstance(val, str), 'category_3_code must be a string.'
        self.security_info['category_3_code'] = val

    @property
    def ticker_info(self):
        return self._ticker_info

    @ticker_info.setter
    def ticker_info(self, tkr_info):
        """ Retrieve a pandas DataFrame containing information about the available time series data.
        
            The index of the DataFrame is the 'series_type_code', which tells us the type of time
            series data. For example, 'RI' is a total return index, 'PI' is a price index, etc.
            Each series_type_code (e.g. 'RI') must be associated with a single ticker. While
            the database itself may contain multiple total return indices for a given security 
            (for example, S&P 500 data found in Bloomberg, and also found in Datastream), the
            user must only provide one of these sets of data to the Security objects. Otherwise,
            the Security object would not know whether to use the Bloomberg ticker or the Datastream
            ticker for the total return index.
                
            The ticker_info DataFrame object allows the user to retrieve the ticker for a given type
            of time series, and then the actual time series data can be obtained using the Database object.
        """
        # Check that the series codes are all unique
        if tkr_info is not None:
            assert isinstance(tkr_info, pd.DataFrame), 'ticker_info must be a pandas DataFrame instance.'
            assert not tkr_info.index.duplicated().any(), 'Each series type may only be associated with one ticker.'
            self._ticker_info = tkr_info
        
    @property
    def ts_db(self):
        """ A Database type object which allows us to retrieve time series information. """
        return self._ts_db
    
    @ts_db.setter
    def ts_db(self, db):
        if hasattr(db, 'get_time_series'):
            if self._ts_db is None:
                self._ts_db = defaultdict(lambda : db)
            else:
                raise ValueError('Cannot set default time series database after it has been initialized.')
        elif db is None:
            self._ts_db = None
        elif isinstance(db, dict):
            if self._ts_db is None:
                if isinstance(db, defaultdict):
                    self._ts_db = db
                else:
                    self._ts_db = defaultdict(lambda : None, db)
            else:
                self._ts_db.update(db)
        else:
            raise ValueError('Unknown database type: {}'.format(db.__class__))

    def copy(self):
        return copy.copy(self)
    
    def deepcopy(self):
        return copy.deepcopy(self)

    def cast(self, class_handle):
        """ Cast from one Security subclass to a different Security subclass. 
        
            For example, given a generic Security object, one could cast it to an AssetIndex by using:
            >> sec.cast('AssetIndex')
            
        """
        if isinstance(class_handle, str):
            class_handle = eval(class_handle)
        return class_handle(security_info=self.security_info.copy(), 
                            ticker_info=self.ticker_info.copy(), 
                            ts_db=self.ts_db)

    def get_tickers(self, series_type_codes):
        """ Retrieve the tickers corresponding to a single or list of series type codes. 
        """
        if isinstance(series_type_codes, str):
            series_type_codes = [series_type_codes]
        return self.ticker_info.loc[series_type_codes].ticker_code.values

    def get_ts_dbs(self, series_type_codes: list):
        """ Retrieve the Database objects corresponding to a single or list of series type codes. 
        
            This function is overkill for the single Security class, but is extremely helpful
            for the Panel class defined in the 'panel' module.
        """        
        if isinstance(series_type_codes, str):
            series_type_codes = [series_type_codes]        
        if self.ts_db is None:
            return [None] * len(series_type_codes)
        else:
            return [self.ts_db[code] for code in series_type_codes]
        
    def get_time_series(self, series_type_codes, frequency='', start=None, end=None, rescale=True, backfill=False):
        """ Obtain the time series data for a single or list of series type codes.
        
            Arguments:
                series_type_codes: (str/list) a single series type code or list of such codes, which indicate
                    the type of time series data (e.g. price return or total return index) that is desired
                frequency: (str) the observation frequency that is desired for the output data. The default
                    value is '', in which case the function returns all available data. Other options are the 
                    pandas frequency codes - D: daily, B: business days, W: weekly, M: monthly, Y: annual.
                start: (datetime/str/Timestamp) the earliest allowable date/time for an observation. If None, 
                    then there is no cutoff. Default is None
                end: (datetime/str/Timestamp) the latest allowable date/time for an observation. If None, 
                    then there is no cutoff. Default is None
                rescale: (bool) whether or not to rescale the time series data, if the database indicates
                    that the rescale factor is not 1. The 'rescale' is often used for interest rates, which
                    often have a rescale factor of 100. Default is True
                backfill: (bool) whether or not to backfill the time series, as specified in the meta data.
                    Default is False.

            Returns a security TimeSeries object which contains both the time series and meta data. This
            type of object is defined in the timeseries module.
        """ 
        sec_info = pd.DataFrame(pd.Series(self.security_info)).T.set_index('sec_code')
        ts, meta = pyfintools.security.helper.get_ts_and_meta(tickers=self.get_tickers(series_type_codes),
                                                         ts_dbs=self.get_ts_dbs(series_type_codes),
                                                         security_info=sec_info,
                                                         ticker_info=self.ticker_info, 
                                                         frequency=frequency, 
                                                         start=start, 
                                                         end=end,
                                                         rescale=rescale,
                                                         backfill=backfill)

        # Construct the time series object
        class_handle = pyfintools.security.helper.get_module_equivalent_class_handle(self, 'single', 'timeseries')
        return class_handle(ts, meta)
    
    def has_time_series(self, series_type_code):
        """ Return True/False if a series type code is available for the security object. """
        return self.ticker_info.size and series_type_code in self.ticker_info.series_type_code.values
    
    def drop_duplicates(self, keep='first'):
        """ Implemented for consistent behavior with the Security Panel class. """
        return self.copy()
        
    def _get_property_value(self, attr, default_val=''):
        """ Retrieve the value for a given attribute name from the security_info data.
        
            This simple function allows us to easily specify a default value if the data is not available.
        """
        if self.security_info is None or attr not in self.security_info:
            return default_val
        else:
            return self.security_info[attr]


class Asset(Security):
    """ An Asset Security object.
    
        Asset objects should have denominated and risk currencies, as they represent real tradable
        financial instruments.
    """
    @property
    def denominated_currency(self):
        return self._get_property_value('denominated_currency', default_val='')

    @denominated_currency.setter
    def denominated_currency(self, val):
        assert isinstance(val, str), 'denominated_currency must be a string.'
        self.security_info['denominated_currency'] = val
            
    @property
    def risk_currency(self):
        return self._get_property_value('risk_currency', default_val='')

    @risk_currency.setter
    def risk_currency(self, val):
        assert isinstance(val, str), 'risk_currency must be a string.'
        self.security_info['risk_currency'] = val
        
    @property
    def risk_region(self):
        return self._get_property_value('risk_region', default_val='')

    @risk_region.setter
    def risk_region(self, val):
        assert isinstance(val, str), 'risk_region must be a string.'
        self.security_info['risk_region'] = val
        
    @property
    def isin(self):
        return self._get_property_value('isin', default_val='')

    @isin.setter
    def isin(self, val):
        assert isinstance(val, str), 'isin must be a string.'
        self.security_info['isin'] = val

        
class Cash(Asset):
    """ A cash Security. """
    pass


class CommoditySpot(Asset):
    """ A commodity spot Security. """
    @property
    def sector(self):
        return self._get_property_value('sector', default_val='')

    @sector.setter
    def sector(self, val):
        assert isinstance(val, str), 'sector must be a string.'
        self.security_info['sector'] = val

    @property
    def sub_sector(self):
        return self._get_property_value('sub_sector', default_val='')

    @sub_sector.setter
    def sub_sector(self, val):
        assert isinstance(val, str), 'sub_sector must be a string.'
        self.security_info['sub_sector'] = val


class Strategy(Asset):
    """ A Strategy Security. 
    
        A Strategy is something that has systematic trading rules associated with it.
        Example 1: rolling an FX forward contract on a monthly basis.
        Example 2: creating a diversified portfolio and rebalancing on a daily basis.
    """
    pass


class Equity(Asset):
    """ An Equity Security, which inherits from Asset objects, as these are tradable instruments."""
    
    @property
    def issuer_name(self):
        return self._get_property_value('issuer_name', default_val='')

    @issuer_name.setter
    def issuer_name(self, val):
        assert isinstance(val, str), 'issuer_name must be a string.'
        self.security_info['issuer_name'] = val


class CommonStock(Equity):
    """ Represents common stocks, and inherits from Asset objects as these are tradable instruments."""
    
    @property
    def domicile(self):
        return self._get_property_value('domicile', default_val='')

    @domicile.setter
    def domicile(self, val):
        assert isinstance(val, str), 'domicile must be a string.'
        self.security_info['domicile'] = val

    @property
    def risk_region(self):
        return self.domicile

    @property
    def sector(self):
        return self._get_property_value('sector', default_val='')

    @sector.setter
    def sector(self, val):
        assert isinstance(val, str), 'sector must be a string.'
        self.security_info['sector'] = val

    @property
    def industry_group(self):
        return self._get_property_value('industry_group', default_val='')

    @industry_group.setter
    def industry_group(self, val):
        assert isinstance(val, str), 'industry_group must be a string.'
        self.security_info['industry_group'] = val

    @property
    def industry(self):
        return self._get_property_value('industry', default_val='')

    @industry.setter
    def industry(self, val):
        assert isinstance(val, str), 'industry must be a string.'
        self.security_info['industry'] = val

    @property
    def sub_industry(self):
        return self._get_property_value('sub_industry', default_val='')

    @sub_industry.setter
    def sub_industry(self, val):
        assert isinstance(val, str), 'sub_industry must be a string.'
        self.security_info['sub_industry'] = val


class PreferredStock(Equity):
    """ Represents preferred stocks, and inherits from Asset objects as these are tradable instruments."""
    pass


class ExchangeTradedEquity(Equity):
    """ Represents ETFs/ETNs, and inherits from Asset objects as these are tradable instruments."""
    
    @property
    def underlier_sec_code(self):
        return self._get_property_value('underlier_sec_code', default_val='')

    @underlier_sec_code.setter
    def underlier_sec_code(self, val):
        assert isinstance(val, str), 'underlier_sec_code must be a string.'
        self.security_info['underlier_sec_code'] = val

    @property
    def leverage(self):
        return self._get_property_value('leverage', default_val=np.nan)

    @leverage.setter
    def leverage(self, val):
        assert isinstance(val, pyfintools.constants.NUMERIC_DATA_TYPES), 'leverage must be a float or integer.'
        self.security_info['leverage'] = float(val)


class ETF(ExchangeTradedEquity):
    """ Represents ETFs, and inherits from Asset objects as these are tradable instruments.

        Exchange Traded Funds (ETFs) have subtley different qualities than ETNs, which is why
        they have their own class.
    """    
    pass


class ETN(ExchangeTradedEquity):
    """ Represents ETNs, and inherits from Asset objects as these are tradable instruments.
    
        Exchange Traded Notes (ETNs) have subtley different qualities than ETFs, which is why
        they have their own class.
    """
    pass


class Bond(Asset):
    """ Represents bonds, and inherits from Asset objects as these are tradable instruments."""
    
    @property
    def issuer_name(self):
        return self._get_property_value('issuer_name', default_val='')

    @issuer_name.setter
    def issuer_name(self, val):
        assert isinstance(val, str), 'issuer_name must be a string.'
        self.security_info['issuer_name'] = val

    @property
    def par_value(self):
        return self._get_property_value('par_value', default_val=np.nan)

    @par_value.setter
    def par_value(self, val):
        assert isinstance(val, pyfintools.constants.NUMERIC_DATA_TYPES), 'par_value must be a float or integer.'
        self.security_info['par_value'] = float(val)

    @property
    def issue_date(self):
        return self._get_property_value('issue_date', default_val='')

    @issue_date.setter
    def issue_date(self, val):
        formatted_val = _format_date_value(val, prop_name='issue_date')
        self.security_info['issue_date'] = formatted_val

    @property
    def maturity_date(self):
        return self._get_property_value('maturity_date', default_val='')

    @maturity_date.setter
    def maturity_date(self, val):
        formatted_val = _format_date_value(val, prop_name='maturity_date')
        self.security_info['maturity_date'] = formatted_val

    @property
    def day_count(self):
        return self._get_property_value('day_count', default_val='')

    @day_count.setter
    def day_count(self, val):
        assert isinstance(val, str), 'day_count must be a string.'
        self.security_info['day_count'] = val

    @property
    def coupon_rate(self):
        return self._get_property_value('coupon_rate', default_val=np.nan)

    @coupon_rate.setter
    def coupon_rate(self, val):
        assert isinstance(val, pyfintools.constants.NUMERIC_DATA_TYPES), 'coupon_rate must be a float or integer.'
        self.security_info['coupon_rate'] = float(val)

    @property
    def coupon_frequency(self):
        return self._get_property_value('coupon_frequency', default_val='')

    @coupon_frequency.setter
    def coupon_frequency(self, val):
        assert isinstance(val, str), 'coupon_frequency must be a string.'
        self.security_info['coupon_frequency'] = val
                                                                

class StraightBond(Bond):
    """ Represents straight bonds, and inherits from Asset objects as these are tradable instruments."""
    pass


class FloatinRateNote(Bond):
    """ Represents floating rate bonds, and inherits from Asset objects as these are tradable instruments."""
    pass


class OriginalIssueDiscount(Bond):
    """ Represents OID bonds, and inherits from Asset objects as these are tradable instruments."""
    pass


class InflationProtectedSecurity(Bond):
    """ Represents IPS, and inherits from Asset objects as these are tradable instruments."""
    pass


class Derivative(Asset):
    """ Represents derivatives, and inherits from Asset objects as these are tradable instruments."""    
    @property
    def expiration_date(self):
        return self._get_property_value('expiration_date', default_val='')

    @expiration_date.setter
    def expiration_date(self, val):
        formatted_val = _format_date_value(val, prop_name='expiration_date')
        self.security_info['expiration_date'] = formatted_val


class Factor(Security):
    """ Represents a factor, e.g. for use in a factor model. """
    pass


class Forward(Derivative):
    """ Represents a forward contract, and inherits from the Derivative class. """
    pass


class Future(Derivative):
    """ Represents a futures contract, and inherits from the Derivative class. """
    pass    
   

class Option(Derivative):
    """ Represents an option contract, and inherits from the Derivative class. """
    
    @property
    def strike(self):
        return self._get_property_value('strike', default_val=np.nan)

    @strike.setter
    def strike(self, val):
        assert isinstance(val, pyfintools.constants.NUMERIC_DATA_TYPES), 'strike must be a float or integer.'
        self.security_info['strike'] = float(val)

    @property
    def exercise_type(self):
        return self._get_property_value('exercise_type', default_val='')

    @exercise_type.setter
    def exercise_type(self, val):
        assert isinstance(val, str), 'exercise_type must be a string.'
        self.security_info['exercise_type'] = val

    @property
    def option_type(self):
        return self._get_property_value('option_type', default_val='')

    @option_type.setter
    def option_type(self, val):
        assert isinstance(val, str), 'option_type must be a string.'
        self.security_info['option_type'] = val


class GenericIndex(Security):
    """ Represents a generic index, e.g. an equity or bond index. """
    
    @property
    def index_provider(self):
        return self._get_property_value('index_provider', default_val='')

    @index_provider.setter
    def index_provider(self, val):
        assert isinstance(val, str), 'index_provider must be a string.'
        self.security_info['index_provider'] = val


class PriceIndex(GenericIndex):
    """ Represents a generic price index, e.g. an equity or bond price index. 
    
        A PriceIndex contains information about the risk currrency and risk region, which would
        not necessarily be relevant for a GenericIndex object.
    """
    
    @property
    def risk_currency(self):
        return self._get_property_value('risk_currency', default_val='')

    @risk_currency.setter
    def risk_currency(self, val):
        assert isinstance(val, str), 'risk_currency must be a string.'
        self.security_info['risk_currency'] = val

    @property
    def risk_region(self):
        return self._get_property_value('risk_region', default_val='')

    @risk_region.setter
    def risk_region(self, val):
        assert isinstance(val, str), 'risk_region must be a string.'
        self.security_info['risk_region'] = val


class EconomicPriceIndex(PriceIndex):
    """ Represents an economic price index, e.g. US GDP.
    
        Economic price indices also contain information about seasonal adjustment factors and 
        the price type (e.g. constant or current prices)
    """
    @property
    def seasonal_adjustment(self):
        return self._get_property_value('seasonal_adjustment', default_val='')

    @seasonal_adjustment.setter
    def seasonal_adjustment(self, val):
        assert isinstance(val, str), 'seasonal_adjustment must be a string.'
        self.security_info['seasonal_adjustment'] = val

    @property
    def price_type(self):
        return self._get_property_value('price_type', default_val='')

    @price_type.setter
    def price_type(self, val):
        assert isinstance(val, str), 'price_type must be a string.'
        self.security_info['price_type'] = val

        
class AssetIndex(PriceIndex):
    """ Represents an asset price index, e.g. MSCI USA.
    
        In addition to a normal price index, an asset price index will also contain information
        about the denominated currency of the series, as well as whether the time series 
        is currency hedged or unhedged.
            
        Note that a hedged and unhedged instance of a time series would be represented by two
        different securities in our framework here.
    """
    @property
    def denominated_currency(self):
        return self._get_property_value('denominated_currency', default_val='')

    @denominated_currency.setter
    def denominated_currency(self, val):
        assert isinstance(val, str), 'denominated_currency must be a string.'
        self.security_info['denominated_currency'] = val

    @property
    def hedging_ratio(self):
        return self._get_property_value('hedging_ratio', default_val=np.nan)

    @hedging_ratio.setter
    def hedging_ratio(self, val):
        assert isinstance(val, pyfintools.constants.NUMERIC_DATA_TYPES), 'hedging_ratio must be a float or integer.'
        self.security_info['hedging_ratio'] = float(val)


class EquityIndex(AssetIndex):
    """ An Equity index Security object. """

    @property
    def market_cap(self):
        return self._get_property_value('market_cap', default_val='')

    @market_cap.setter
    def market_cap(self, val):
        assert isinstance(val, str), 'market_cap must be a string.'
        self.security_info['market_cap'] = val

    @property
    def factor_style(self):
        return self._get_property_value('factor_style', default_val='')

    @factor_style.setter
    def factor_style(self, val):
        assert isinstance(val, str), 'factor_style must be a string.'
        self.security_info['factor_style'] = val

    @property
    def gics_sector(self):
        return self._get_property_value('gics_sector', default_val='')

    @gics_sector.setter
    def gics_sector(self, val):
        assert isinstance(val, str), 'gics_sector must be a string.'
        self.security_info['gics_sector'] = val


class BondIndex(AssetIndex):
    """ A Bond index Security object. """

    @property
    def issuer_segment(self):
        return self._get_property_value('issuer_segment', default_val='')

    @issuer_segment.setter
    def issuer_segment(self, val):
        assert isinstance(val, str), 'issuer_segment must be a string.'
        self.security_info['issuer_segment'] = val

    @property
    def ratings_segment(self):
        return self._get_property_value('ratings_segment', default_val='')

    @ratings_segment.setter
    def ratings_segment(self, val):
        assert isinstance(val, str), 'ratings_segment must be a string.'
        self.security_info['ratings_segment'] = val

    @property
    def maturity_bucket(self):
        return self._get_property_value('maturity_bucket', default_val='')

    @maturity_bucket.setter
    def maturity_bucket(self, val):
        assert isinstance(val, str), 'maturity_bucket must be a string.'
        self.security_info['maturity_bucket'] = val
    
    @property
    def inflation_protected(self):
        return self._get_property_value('inflation_protected', default_val=False)

    @inflation_protected.setter
    def inflation_protected(self, val):
        assert isinstance(val, bool), 'inflation_protected must be True/False.'
        self.security_info['inflation_protected'] = val

    
class CommodityIndex(AssetIndex):
    """ A Commodity index Security object. """   

    @property
    def sector(self):
        return self._get_property_value('sector', default_val='')

    @sector.setter
    def sector(self, val):
        assert isinstance(val, str), 'sector must be a string.'
        self.security_info['sector'] = val

    @property
    def sub_sector(self):
        return self._get_property_value('sub_sector', default_val='')

    @sub_sector.setter
    def sub_sector(self, val):
        assert isinstance(val, str), 'sub_sector must be a string.'
        self.security_info['sub_sector'] = val


class RealEstateIndex(AssetIndex):
    """ A Real Estate index Security object. """
    @property
    def segment(self):
        return self._get_property_value('segment', default_val='')

    @segment.setter
    def segment(self, val):
        assert isinstance(val, str), 'segment must be a string.'
        self.security_info['segment'] = val


class HedgeFundIndex(AssetIndex):
    """ A Hedge fund index Security object. """

    @property
    def strategy(self):
        return self._get_property_value('strategy', default_val='')

    @strategy.setter
    def strategy(self, val):
        assert isinstance(val, str), 'strategy must be a string.'
        self.security_info['strategy'] = val

    @property
    def substrategy(self):
        return self._get_property_value('substrategy', default_val='')

    @substrategy.setter
    def substrategy(self, val):
        assert isinstance(val, str), 'substrategy must be a string.'
        self.security_info['substrategy'] = val

    @property
    def weighting(self):
        return self._get_property_value('weighting', default_val='')

    @weighting.setter
    def weighting(self, val):
        assert isinstance(val, str), 'weighting must be a string.'
        self.security_info['weighting'] = val


class Rates(Security):
    """ A Rates Security object. 
    
        This class is useful for working with generic rates, and is used as
        a base class for FX and yield time series.
    """

    def __init__(self, security_info=None, ticker_info=None, ts_db=None):
        super().__init__(security_info=security_info, ticker_info=ticker_info, ts_db=ts_db)
        self.security_info['tenor_in_years'] = self._get_tenor_in_years()
            
    @property
    def tenor(self):
        return self._get_property_value('tenor', default_val='')

    @tenor.setter
    def tenor(self, val):
        assert isinstance(val, str), 'tenor must be a string.'
        self.security_info['tenor'] = val
        self.security_info['tenor_in_years'] = self._get_tenor_in_years()

    @property
    def tenor_in_years(self):
        return self._get_property_value('tenor_in_years', default_val=np.nan)

    @tenor_in_years.setter
    def tenor_in_years(self, val):
        raise NotImplementedError('Need to implement setter function for tenor_in_years.')

    def _get_tenor_in_years(self):
        if self.tenor == pyfintools.security.constants.TENOR_SPOT:
            return 0
        elif self.tenor:
            return 1 / pyfintools.tools.freq.get_periods_per_year(self.tenor)
        else:
            return np.nan

    @property
    def index_provider(self):
        return self._get_property_value('index_provider', default_val='')

    @index_provider.setter
    def index_provider(self, val):
        assert isinstance(val, str), 'index_provider must be a string.'
        self.security_info['index_provider'] = val


class InterestRate(Rates):
    """ A Security class for working with interest rates. """

    @property
    def denominated_currency(self):
        return self._get_property_value('denominated_currency', default_val='')

    @property
    def risk_currency(self):
        return self._get_property_value('risk_currency', default_val='')

    @risk_currency.setter
    def risk_currency(self, val):
        assert isinstance(val, str), 'risk_currency must be a string.'
        self.security_info['risk_currency'] = val

    @property
    def risk_region(self):
        return self._get_property_value('risk_region', default_val='')

    @risk_region.setter
    def risk_region(self, val):
        assert isinstance(val, str), 'risk_region must be a string.'
        self.security_info['risk_region'] = val

    @property
    def issuer_name(self):
        return self._get_property_value('issuer_name', default_val='')

    @issuer_name.setter
    def issuer_name(self, val):
        assert isinstance(val, str), 'issuer_name must be a string.'
        self.security_info['issuer_name'] = val

    @property
    def issuer_segment(self):
        return self._get_property_value('issuer_segment', default_val='')

    @issuer_segment.setter
    def issuer_segment(self, val):
        assert isinstance(val, str), 'issuer_segment must be a string.'
        self.security_info['issuer_segment'] = val

    @property
    def ratings_segment(self):
        return self._get_property_value('ratings_segment', default_val='')

    @ratings_segment.setter
    def ratings_segment(self, val):
        assert isinstance(val, str), 'ratings_segment must be a string.'
        self.security_info['ratings_segment'] = val

    @property
    def rate_type(self):
        return self._get_property_value('rate_type', default_val='')

    @rate_type.setter
    def rate_type(self, val):
        assert isinstance(val, str), 'rate_type must be a string.'
        self.security_info['rate_type'] = val
    
    @property
    def inflation_protected(self):
        return self._get_property_value('inflation_protected', default_val=False)

    @inflation_protected.setter
    def inflation_protected(self, val):
        assert isinstance(val, bool), 'inflation_protected must be True/False.'
        self.security_info['inflation_protected'] = val


class FX(Rates):
    """ A Security class for working with FX rates. """

    def __init__(self, security_info=None, ticker_info=None, ts_db=None):
        super().__init__(security_info=security_info, ticker_info=ticker_info, ts_db=ts_db)
        self.security_info['currency_pair'] = self._get_currency_pair()
        
    @property
    def base_currency(self):
        return self._get_property_value('base_currency', default_val='')

    @base_currency.setter
    def base_currency(self, val):
        assert isinstance(val, str), 'base_currency must be a string.'
        self.security_info['base_currency'] = val
        self.security_info['currency_pair'] = self._get_currency_pair()

    @property
    def quote_currency(self):
        return self._get_property_value('quote_currency', default_val='')

    @quote_currency.setter
    def quote_currency(self, val):
        assert isinstance(val, str), 'quote_currency must be a string.'
        self.security_info['quote_currency'] = val
        self.security_info['currency_pair'] = self._get_currency_pair()

    @property
    def currency_pair(self):
        return self._get_property_value('currency_pair', default_val='')

    @currency_pair.setter
    def currency_pair(self):
        raise NotImplementedError('Until implemented, you can only set the base and quote currencies separately.')

    def _get_currency_pair(self):
        return '{base}/{quote}'.format(base=self.base_currency, quote=self.quote_currency)


class PPP(FX):
    """ A Security class for working with Purchasing Power Parity (PPP) series. """

    pass
    

def from_metadata(security_info, ticker_info, ts_db=None):
    """ A factory function for creating a Security object from the meta data and Database object.
    
        From the meta data (e.g. the category 1-3 codes), this function determines the appropriate
        Security class for the given instrument, and returns the initialized class.
    """ 
    return pyfintools.security.helper.get_security(security_info, ticker_info, ts_db=ts_db)

def _format_date_value(val, prop_name):
    """ A helper function for correctly formatting the meta data pertaining to dates/times.
    """
    
    if val is None:
        return None
    elif isinstance(val, str):
        try:
            return pd.Timestamp(val).date()
        except:
            raise ValueError(f'Unsupported date format for {prop_name}: {val}')
    elif isinstance(val, (datetime.datetime, pd.Timestamp)):
        return val.date()
    elif isinstance(val, datetime.date):
        return val
    else:
        raise ValueError(f'Unsupported date format for {prop_name}: {val}')
