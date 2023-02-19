""" Provides unified access to time series and meta data for multiple securities.

    This module is intended to provide a set of classes that allow the user
    to easily switch between time series and meta data for a list of financial
    securities. The base class Panel is then inherited by all subclasses,
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
            series data are available for the given securities
        ts_db: a Database object (defined in the 'database' module), that gives access
            to time series data.

    The numerous classes here inherit from the base class Panel, and include additional
    variables and behaviours as necessary.
"""


import copy
import numpy as np
import pandas as pd
from collections import Iterable

import pyfintools.tools.freq
import pyfintools.security.helper

from pyfintools.constants import NUMERIC_DATA_TYPES
from pyfintools.security.constants import TENOR_SPOT, DEFAULT_FX_HEDGING_FREQUENCY, DEFAULT_CROSS_CURRENCY

class Panel(object):
    """ A class that offers unified access to time series and meta data for a list of securities.
        
        This base class is intended to be inherited by subclasses that allow the user
        to easily switch between time series and meta data for a single financial
        security.

        Attributes:
            object_list: a list of single Security objects (from the 'single' module). To work with 
                this object_list, the Panel object includes many familiar list methods such as 
                'append', 'extend', 'pop', 'index', etc.         
                Objects inheriting from the Panel class have functionality similar to a Python list object.
            allow_duplicates: (bool) whether or not to allow duplicate Security objects in the Panel.
            security_info: (DataFrame) contains key meta data about the security
            ticker_info: (DataFrame) constains information about which 
                    types of time series data are available for the given security
            ts_db: (Database) a Database object that provides access to time series information
            is_tradable: (bool) describes whether the Panel is a tradable financial instrument 
                    (e.g. True for a single bond, but False for a bond index)
            sec_code: (str) a unique security code
            name: (str) the name of the security
            category_1_code: (str) the category 1 code of the security (e.g. EQ, BD, FX, etc)
            category_2_code: (str) the category 2 code of the security (e.g. FXSpot, EQI, etc.)
            category_3_code: (str) the category 3 code of the security
    """

    def __init__(self, object_list=None, allow_duplicates=False):
        """ Initialize Security object.
        
            Arguments:
                object_list: a list of single Security objects (from the 'single' module). To work with 
                    this object_list, the Panel object includes many familiar list methods such as 
                    'append', 'extend', 'pop', 'index', etc.         
                    Objects inheriting from the Panel class have functionality similar to a Python list object.
                allow_duplicates: (bool) whether or not to allow duplicate Security objects in the Panel.

        """        
        super().__init__()
        self.allow_duplicates = allow_duplicates
        self.object_list = []        
        if object_list is not None:
            self.extend(object_list)

    def _constructor(self, object_list=None, allow_duplicates=False):
        return self.__class__(object_list=object_list, allow_duplicates=allow_duplicates)

    @property
    def size(self):
        return len(self.object_list)

    def copy(self):
        return copy.copy(self)
    
    def deepcopy(self):
        """ Perform a deep copy of the Panel object.
            We do not copy the database handles when we create a deep copy. 
            """
        return self._deepcopy_new_class()
    
    def get_securities(self, target_sec_codes):
        """Select a subset of the security panel from a list of security codes.
           The securities in the resulting panel will be ordered the same as the input ordering.
           """
        indices = []
        all_sec_codes = self.sec_code
        for code in target_sec_codes:
            if code not in target_sec_codes:
                raise ValueError(f'Security {code} is not in the panel.')
            else:
                indices.append(all_sec_codes.index(code))            
        return self[indices]
    
    def get_tickers(self, series_type_codes):
        """ Get a list of tickers for all the series type codes and for each Security in the Panel. 
        """
        # Make sure the input is a list
        if isinstance(series_type_codes, str):
            series_type_codes = [series_type_codes]

        # Get nested list of lists (different series types for each security)        
        tickers = [sec.get_tickers(series_type_codes) for sec in self]
        return [tkr for series_tickers in tickers for tkr in series_tickers]
        
    def get_ts_dbs(self, series_type_codes):
        """ Get a list of time series dbs for all the series type codes and for each Security in the Panel.
        """
        # Make sure the input is a list        
        if isinstance(series_type_codes, str):
            series_type_codes = [series_type_codes]

        # Get nested list of lists (different series types for each security)
        ts_dbs = [sec.get_ts_dbs(series_type_codes) for sec in self]
        return [db for series_dbs in ts_dbs for db in series_dbs]
        
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
        if not self.object_list:
            ts, meta = pd.DataFrame(), pd.DataFrame()
        else:
            ts, meta = pyfintools.security.helper.get_ts_and_meta(tickers=self.get_tickers(series_type_codes),
                                                             ts_dbs=self.get_ts_dbs(series_type_codes),
                                                             security_info=self.security_info, 
                                                             ticker_info=self.ticker_info, 
                                                             frequency=frequency, 
                                                             start=start, 
                                                             end=end,
                                                             rescale=rescale,
                                                             backfill=backfill)

        # Construct the time series object
        class_handle = pyfintools.security.helper.get_module_equivalent_class_handle(self, 'panel', 'timeseries')
        return class_handle(ts, meta)
    
    def has_time_series(self, series_type_code):
        """ Tells us whether or not the time series data is available for a specific series type code. """
        return np.all([sec.has_time_series(series_type_code) for sec in self.object_list])

    def append(self, _object):
        """ Append a new Security onto the object_list of the Panel.
        """
        if not self.allow_duplicates and _object in self:
            raise ValueError('This panel does not allow duplicate securities to be added.')
        else:
            self.object_list.append(_object)

    def extend(self, iterable):
        """ Add several new Security objects onto the object_list of the Panel.
        
            
        """        
        if not self.allow_duplicates and any([obj in self for obj in iterable]):
            raise ValueError('This panel does not allow duplicate securities to be added.')
        else:
            self.object_list.extend(iterable)
    
    def count(self, _sec_code):
        """ Return the number of times a particular security code (sec_code) appears in the object list.
        """        
        return self.sec_code.count(_sec_code)
        
    def index(self, _sec_code, start=0, end=None):
        """ Find the location of a security code within the object list. 
        
            Functionality is similar to the 'index' method for 'list'-type objects in Python. 
        """
        if end is not None:
            return self.sec_code.index(_sec_code, start, end)
        else:
            return self.sec_code.index(_sec_code, start)
    
    def pop(self, idx=-1):
        """ Pop the Security at location 'idx' from the object list. 
        
            Functionality is similar to the 'pop' method for 'list'-type objects in Python. 
        """        
        self.object_list.pop(idx)
    
    def remove(self, _sec_code):
        """ Remove any Security objects from the Panel if their sec_code matches the input. """
        idx = self.sec_code.index(_sec_code)
        self.object_list.pop(idx)
        
    def reverse(self):
        """ Reverse the order of the Security objects in object_list. """
        self.object_list.reverse()
        
    def select(self, attrib_name, target_values):
        """ Select a subset of Security objects from the object_list if their value
            of the attribute 'attrib_name' match the target_values.
            
            Typical usage:
                Assume we have a Panel instance called 'sec_panel'. Then, we can run:
                >> sec_panel.select('category_1_code', 'EQI')
                
                The result would be a new Panel object with a subset of the original
                Panel object, such that all of the new objects had category_1_code == "EQI"
        """
        attrib_vals = self.__getattribute__(attrib_name)
        if not isinstance(target_values, Iterable) or isinstance(target_values, str):
            target_values = [target_values]

        locations = dict()
        for j, val in enumerate(attrib_vals):
            if val in target_values:
                if val not in locations:
                    locations[val] = j
                else:
                    raise ValueError(f'Multiple securities with the same attribute value ({val}) are not permitted.')

        if len(target_values) != len(locations):
            missing = set(target_values) - set(locations.keys())
            raise ValueError('Missing some target values: {}'.format(list(missing)))
        else:
            idx = [locations[val] for val in target_values]
            return self[idx]

    def sort_values(self, attrib_name, reverse=False):
        """ Sort the objects in the object_list according to their values of 'attrib_name'.
        
            Arguments:
                attrib_name: (str) the name of the property by which to sort the objects
                reverse: (bool) whether to reverse the order of the sorted list. Default is False
        """
        attrib_vals = self.__getattribute__(attrib_name)
        idx = list(np.argsort(attrib_vals))
        if reverse:
            idx = idx[::-1]
        return self[idx]

    def query(self, expr, inplace=False, **kwargs):
        """ Run a SQL-type query on the properties in the Panel object.
        
            This method is inpsired by the pandas DataFrame method 'query', so 
            you can look at that documentation for more info.
            
            Arguments:
                expr: (str) an SQL-type query string that specifies which of the
                    objects in the object_list we want to keep.
                inplace: (bool) whether to perform the query on the object in place.
                    Default is False, and in-place querying is not yet supported.
        """
        assert not inplace, 'In place querying is not supported.'
        sec_info = self.security_info.query(expr, inplace=False, **kwargs)
        return self.apply_filter('sec_code', sec_info.index.values)
    
    def apply_filter(self, attrib_name, target_values):
        """ Filter the objects to obtain a new Panel object with a subset of original objects.
            
            Arguments:
                attrib_name: (str) the name of an attribute/variable of the Panel object
                target_values: (list/str/float/int) the single value or list of values
                    that the retained objects must have
                    
            Typical Usage:
                # To keep all Securities that have USD as their denominated currency:
                >> sec_panel.apply_filter('denominated_currency', 'USD')
                
                # To keep all Securities that have EUR or JPY as their risk currency:
                >> sec_panel.apply_filter('risk_currency', ['EUR', 'JPY'])
        """
        if isinstance(target_values, (list, set, np.ndarray)):
            return self[[x in target_values for x in self.__getattribute__(attrib_name)]]
        else:
            return self[[x == target_values for x in self.__getattribute__(attrib_name)]]

    def drop(self, attrib_name, target_values):
        """ Drop specified objects to obtain a new Panel object with a subset of original objects.
            
            Arguments:
                attrib_name: (str) the name of an attribute/variable of the Panel object
                target_values: (list/str/float/int) the single value or list of values
                    that the retained objects must have
                    
            Typical Usage:
                # To remove all Securities that have USD as their denominated currency:
                >> sec_panel.drop('denominated_currency', 'USD')
                
                # To remove all Securities that have EUR or JPY as their risk currency:
                >> sec_panel.drop('risk_currency', ['EUR', 'JPY'])
        """        
        if isinstance(target_values, (list, set)):
            return self[[x not in target_values for x in self.__getattribute__(attrib_name)]]
        else:
            return self[[x != target_values for x in self.__getattribute__(attrib_name)]]

    def duplicated(self, keep='first'):
        """ Returns a list of True/False specifying whether each Security in the object_list is duplicated. """
        sec_codes = pd.Index(self.sec_code)
        return sec_codes.duplicated(keep=keep)
    
    def drop_duplicates(self, keep='first'):
        """ Drop duplicates from the object_list. """
        idx_dup = self.duplicated(keep=keep)
        return self[~idx_dup]

    def cast(self, class_handle):
        """ Cast from one Panel subclass to a different Panel subclass. 
        
            Arguments:
                class_handle (str): the name of the new class to which we cast.
                
            Typical Usage:
                # Cast the current Panel object to an AssetIndex Panel.
                >> sec_panel.cast('AssetIndex')            
        """
        return self._deepcopy_new_class(class_handle)

    def _deepcopy_new_class(self, panel_class_handle=None):
        """ Create a deep copy of the Panel.
        
            Try to align the Security object types in object_list with the Panel object class.
        """
        if panel_class_handle is None:
            panel_class_handle = self.__class__
        elif isinstance(panel_class_handle, str):
            panel_class_handle = eval(panel_class_handle)
            
        new_object_list = []
        for sec in self.object_list:
            sec_class_handle = panel_class_handle.__name__
            if sec_class_handle == 'Panel':
                sec_class_handle = 'Security'

            new_sec = sec.cast(sec_class_handle)
            new_object_list.append(new_sec)

        return panel_class_handle(object_list=new_object_list, allow_duplicates=self.allow_duplicates)

    def __add__(self, other):
        """ Overload the '+' operator to allow combining two Panel objects
            
            Typical usage:
                # Starting with Panel obj_1 and Panel obj_2, we can combine them via:
                >> obj_1 + obj_2                
            """
        obj_list_1 = self.object_list
        obj_list_2 = other.object_list
        class_handle = type(self)
        if not isinstance(self, other.__class__):
            raise ValueError('Combine panels is only supported if they are of the same class.')
        elif self.allow_duplicates != other.allow_duplicates:
            raise ValueError('Panels cannot be combined if "allow_duplicates" is not a common value.')
        else:
            return class_handle(obj_list_1 + obj_list_2, self.allow_duplicates)
        
    def __iter__(self):
        """ Make an iterator from the object_list. """
        return iter(self.object_list)
    
    def __reversed__(self):
        return reversed(self.object_list)
    
    def __contains__(self, _sec_code):
        return _sec_code in self.sec_code
    
    def __getitem__(self, val):
        if isinstance(val, int):
            # Return a single Security object if an integer is provided
            return self.object_list[val]
        elif isinstance(val, (float, np.float32, str)):
            raise ValueError('Unsupported input type: {}'.format(val.__class__))
        elif isinstance(val, (np.ndarray, list)) and list(val) == []:
            return self._constructor(object_list=[])
        else:
            # ...otherwise return a new security Panel object
            if isinstance(val, np.ndarray):
                if val.dtype == 'bool':
                    val = [bool(x) for x in val]
                else:
                    val = [x for x in val]
                
            if isinstance(val, list):
                if all([isinstance(x, bool) for x in val]):
                    # Create a mask for boolean array                  
                    assert len(val) == len(self.object_list), 'Dimsional mismatch in mask.'
                    object_sub_list = [self.object_list[j] for j in range(len(val)) if val[j]]                    
                else:
                    object_sub_list = [self.object_list[idx] for idx in val]
            elif isinstance(val, slice):
                object_sub_list = self.object_list[val]
            else:
                raise ValueError('Unknown input type.')
            return self._constructor(object_list=object_sub_list, allow_duplicates=self.allow_duplicates)

    def _get_property_value(self, attr, default_val=''):
        if self.object_list and not hasattr(self[0], attr):
            return [default_val] * len(self.object_list)
        else:
            return [obj.__getattribute__(attr) for obj in self.object_list]
    
    @property
    def sec_code(self):
        return self._get_property_value('sec_code')
    
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
    def security_info(self):
        return pd.DataFrame.from_dict(self._get_property_value('security_info')).set_index('sec_code')
    
    @property
    def ticker_info(self):
        tkr_info_list = self._get_property_value('ticker_info')
        return pd.concat([info.set_index('ticker_code') for info in tkr_info_list], axis=0)        

    
class Asset(Panel):
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
    
    
class Cash(Asset):
    pass
    

class CommoditySpot(Asset):
    """ A commodity spot Security Panel. """
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
        return self._get_property_value('coupon_frequency')


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

    @property
    def underlier_sec_code(self):
        return self._get_property_value('underlier_sec_code', '')


class Factor(Panel):
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


class GenericIndex(Panel):
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
        return self._get_property_value('hedging_ratio', np.nan)


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
    def strategy(self):
        return self._get_property_value('sector')

    @property
    def strategy(self):
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


class Rates(Panel):
    @property
    def tenor(self):
        return self._get_property_value('tenor')
    
    @property
    def tenor_in_years(self):    
        return self._get_property_value('tenor_in_years', np.nan)

    @property
    def index_provider(self):
        return self._get_property_value('index_provider')


class InterestRate(Rates):
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

    def get_fx_rates(self, series_type_code, target_ccy_pairs, 
                     target_tenors=TENOR_SPOT,
                     cross_currency=DEFAULT_CROSS_CURRENCY,
                     frequency='', start=None, end=None, backfill=False):
        """ Function to retrieve FX time series for target currency pairs and tenors.
        
            get_fx_rates(self, series_type_code, target_ccy_pairs, **kwargs)
            
            Arguments:
                series_type_code: which series type to use for the underlying FX data
                target_ccy_pairs: a list or string of the currency pair(s) for which data is required.
                target_tenors: a list or string of the tenor(s) for which data is required. 
                               Default tenor is the 'spot' value.
                cross_currency: if no exchange rate is found corresponding to a target currency pair, then
                          the code will try to construct it by going through the cross currency.
                          For example, to make EUR/GBP, we can go through USD with EUR/USD and USD/GBP
                frequency: the frequency for which the time series should be sampled.
                start: the earliest sampled date/time
                end: the latest sampled date/time
                backfill: True/False - whether or not to backfill the time series, using rules from the meta database.
            
            Returns:
                Security TimeSeries object, with columns dual-indexed by the target currency pairs AND the target tenors.
        """
        # Get the correctly formatted tenor and currency pair information from the input arguments
        target_ccy_pairs, target_tenors = pyfintools.security.helper.format_ccy_pair_and_tenor_info(
                                                                target_ccy_pairs, target_tenors)

        # Create an instance of the FXHelper to assist with exchange rate construction
        fx_helper = pyfintools.security.helper.FXHelper(base_currency=self.base_currency,
                                                   quote_currency=self.quote_currency,
                                                   labels=self.sec_code,
                                                   tenor=self.tenor,
                                                   cross_currency=cross_currency)

        # Get the instructions on how to construct the target FX rates
        instructions, req_labels = fx_helper.get_ccy_instructions(target_ccy_pairs, target_tenors)
        
        # Get the underlier time series needed for construction
        fx_sub_panel = self.select('sec_code', req_labels)
        underlier_ts = fx_sub_panel.get_time_series(series_type_code, frequency=frequency,
                                                    start=start, end=end, backfill=backfill)

        # Use the method on the TimeSeries security object to get the FX rates object
        return underlier_ts.get_fx_rates(target_ccy_pairs, target_tenors,
                                         cross_currency=cross_currency)

    def convert_currency(self,
                         series_type_code,
                         asset_ts,
                         to_currency,
                         hedging_ratio,
                         hedging_frequency=DEFAULT_FX_HEDGING_FREQUENCY,
                         cross_currency=DEFAULT_CROSS_CURRENCY,
                         backfill=False):
        assert isinstance(to_currency, str), "Argument 'to_currency' must be a string."
        
        # Get the target currency pairs
        target_ccy_pairs = [f'{bc}/{to_currency}' for bc in asset_ts.denominated_currency]

        # Include all tenors with frequencies higher than the hedging_frequency
        target_freq = pyfintools.tools.freq.get_periods_per_year(hedging_frequency)
        target_tenors = []
        for tnr in set(self.tenor):
            if pyfintools.tools.freq.get_periods_per_year(tnr) >= target_freq:
                target_tenors.append(tnr)

        # Get the FX rates needed for currency conversion
        fx_rates = self.get_fx_rates(series_type_code, 
                                     target_ccy_pairs=target_ccy_pairs, 
                                     target_tenors=TENOR_SPOT,
                                     frequency=pd.infer_freq(asset_ts.index), 
                                     start=asset_ts.index[0], 
                                     end=asset_ts.index[-1],
                                     cross_currency=cross_currency,
                                     backfill=backfill)

        # Use the TimeSeries security object methods to do the FX conversion
        return fx_rates.convert_currency(asset_ts,
                                         to_currency=to_currency,
                                         hedging_ratio=hedging_ratio,
                                         hedging_frequency=hedging_frequency,
                                         cross_currency=cross_currency)

    def get_fx_forward_returns(self, series_type_code, long_currency, short_currency,
                               roll_frequency=DEFAULT_FX_HEDGING_FREQUENCY,
                               cross_currency=DEFAULT_CROSS_CURRENCY,
                               frequency='', start=None, end=None, backfill=False):
        """ Function to get the calculated rolling returns for long/short currency pairs.
            
            Arguments:
                series_type_code: which series type to use for the underlying FX data
                long_currency: list/str - the currency(s) to go long
                short_currency: list/str - the currency(s) to go short
                roll_frequency: list - the frequency for rolling the forward contracts
                cross_currency: if no exchange rate is found corresponding to a target currency pair, then
                          the code will try to construct it by going through the cross currency.
                          For example, to make EUR/GBP, we can go through USD with EUR/USD and USD/GBP
                frequency: the frequency for which the time series should be sampled.
                start: the earliest sampled date/time
                end: the latest sampled date/time
                backfill: True/False - whether or not to backfill the time series, using rules from the meta database.
            
            Returns:
                Security TimeSeries object, with columns dual-indexed by the target currency pairs AND the target tenors.
        """
        if roll_frequency.lower() != '1m':
            raise NotImplementedError('Rolling forwards is currently only supported for 1-month roll frequency.')
        else:
            required_fwd_tenors = [roll_frequency]

        # Get the required currency pairs
        currency_pairs = pyfintools.security.helper.get_currency_pairs_from_inputs(long_currency, short_currency)
        
        # Get the spot rates
        spot_rates = self.get_fx_rates(series_type_code, target_ccy_pairs=currency_pairs,
                                       target_tenors=TENOR_SPOT, cross_currency=cross_currency,
                                       frequency=frequency, start=start, end=end, backfill=backfill)
        
        # Get the forward rates
        fwd_rates = self.get_fx_rates(series_type_code, target_ccy_pairs=currency_pairs,
                                      target_tenors=required_fwd_tenors, cross_currency=cross_currency,
                                      frequency=frequency, start=start, end=end, backfill=backfill)

        # Join the time series into a single object
        fx_rates = spot_rates.merge(fwd_rates)
        
        # Get the correctly formatted tenor and currency pair information from the input arguments
        return fx_rates.get_fx_forward_returns(long_currency=long_currency, short_currency=short_currency, 
                               roll_frequency=roll_frequency, cross_currency=cross_currency)
    
    def get_hedging_proceeds(self,
                             series_type_code,
                             to_currency, 
                             from_currency, 
                             hedging_frequency=DEFAULT_FX_HEDGING_FREQUENCY, 
                             cross_currency=DEFAULT_CROSS_CURRENCY,
                             frequency='', start=None, end=None, backfill=False):
        return self.get_fx_forward_returns(series_type_code, 
                                           long_currency=to_currency, short_currency=from_currency,
                                           roll_frequency=hedging_frequency, cross_currency=cross_currency,
                                           frequency=frequency, start=start, end=end, backfill=backfill)


class PPP(FX):
    pass
    
    
# Define Factory method for creating new Security Panel objects from DataFrames
def from_metadata(security_info, ticker_info, ts_db=None, allow_duplicates=None):
    object_list = []
    uniq_sec_codes = set(security_info.sec_code)
    for sec_code in uniq_sec_codes:
        sec_info = security_info.query(f"sec_code == '{sec_code}'")
        tkr_info = ticker_info.query(f"sec_code == '{sec_code}'")        
        sec = pyfintools.security.helper.get_security(sec_info, tkr_info, ts_db=ts_db)
        object_list.append(sec)
    return from_securities(object_list, allow_duplicates=allow_duplicates)

# Define Factory method for creating new Security Panel objects from DataFrames
def from_securities(object_list, allow_duplicates=None):
    # Get the category codes for the security objects
    cat_1_codes = [obj.category_1_code for obj in object_list]
    cat_2_codes = [obj.category_2_code for obj in object_list]
    class_name = pyfintools.security.helper.get_security_panel_name(cat_1_codes, cat_2_codes, 
                                                               default_class_name='Panel')
    class_handle = eval(class_name)    
    return class_handle(object_list=object_list, allow_duplicates=allow_duplicates)
