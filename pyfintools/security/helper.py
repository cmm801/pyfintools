from collections import defaultdict
import importlib
import pandas as pd
import numpy as np

import pyfintools.tools.fts
import pyfintools.security.constants as constants


def get_module_equivalent_class_handle(obj, from_module, to_module):
    """ Translate classes from one security module to another. For example, get panel object handle
           equivalent to a Security or TimeSeries object. """
    module_base_map = {'single' : 'Security', 'panel' : 'Panel', 'timeseries' : 'TimeSeries' }
    
    # Get the class sub name (the last piece of the dotted module path name)
    class_sub_name = obj.__class__.__name__
    if class_sub_name == module_base_map[from_module]:
        class_sub_name = module_base_map[to_module]
    class_handle = getattr(importlib.import_module('pyfintools.security.timeseries'), class_sub_name)
    return class_handle
    
def get_ts_and_meta(tickers, ts_dbs, security_info, ticker_info, 
                    frequency='', start=None, end=None, rescale=True, backfill=False):
    if len(set(tickers)) != len(list(tickers)):
        raise ValueError('Tickers must be unique.')

    # First, get the time series
    ts = get_time_series_from_db(tickers, ts_dbs, frequency=frequency, start=start, end=end, backfill=backfill)

    # Then extract the meta data
    if ticker_info.index.name is None or ticker_info.index.name != 'ticker_code':
        ticker_info = ticker_info.set_index('ticker_code', drop=False)
    tkr_info = ticker_info.loc[ts.columns]
    sec_info = security_info.loc[tkr_info.sec_code]
    sec_info.index = tkr_info.index
    meta = tkr_info.merge(sec_info, left_index=True, right_index=True).dropna(how='all', axis=1).T
    meta = meta[ts.columns]
    if 'ticker_code' not in meta.index:
        meta.loc['ticker_code'] = ts.columns
    
    # Convert any date columns to datetime
    idx_dt = meta.loc['ts_type'].values == pyfintools.tools.fts.TS_TYPE_DATES
    if np.any(idx_dt):
        ts.iloc[:,idx_dt] = ts.iloc[:,idx_dt].astype(np.datetime64)
    
    # Rescale the time series if necessary
    if rescale:
        # Don't rescale datetime columns
        rescale_values = meta.loc['rescale_factor'].values[~idx_dt].astype(float)
        ts.iloc[:,~idx_dt] = ts.iloc[:,~idx_dt] / rescale_values
    return ts, meta

def get_time_series_from_db(tickers, ts_dbs, frequency='', start=None, end=None, backfill=False):
    assert not any([db is None for db in ts_dbs]), 'No time series database has been specified.'
    assert len(tickers) == len(ts_dbs), 'Dimensional mismatch: tickers and dbs must have same length.'        
    db_map = dict()
    tb_tickers = defaultdict(list)        
    for j, tkr in enumerate(tickers):
        db = ts_dbs[j]
        db_map[db.uid] = db
        tb_tickers[db.uid].append(tkr)

    # Get time series for each database separately
    ts_panels = []
    for uid, db in db_map.items():
        sub_ts = db.get_time_series(tb_tickers[uid], frequency=frequency, start=start, end=end, backfill=backfill)
        ts_panels.append(sub_ts)

    # Combine time series into single object
    ts = pd.concat(ts_panels, sort=False)
    
    # Return a time series with columns ordered as they appeared in the original request
    return ts[tickers]        

# Factor design pattern for retrieving security objects
def get_security(security_info, ticker_info, ts_db=None):
    # Get the class name
    cat_1_code = security_info['category_1_code']
    cat_2_code = security_info['category_2_code']
    class_sub_name = get_security_class_name(cat_1_code, cat_2_code)
    if class_sub_name is None:
        raise ValueError('Unknown security class for category codes: {} {}'.format(cat_1_code, cat_2_code))
    
    class_handle = getattr(importlib.import_module('pyfintools.security.single'), class_sub_name)
    return class_handle(security_info=security_info, ticker_info=ticker_info, ts_db=ts_db)

def get_security_panel_name(cat_1_codes, cat_2_codes, default_class_name):
    # Get the category codes for the security objects
    cat_1_code = cat_1_codes[0] if len(list(set(cat_1_codes))) == 1 else ''
    cat_2_code = cat_2_codes[0] if len(list(set(cat_2_codes))) == 1 else ''

    # Get the name of the corresponding security panel
    class_sub_name = get_security_class_name(cat_1_code, cat_2_code)
    if class_sub_name is None:
        class_sub_name = default_class_name
    return class_sub_name

def get_required_security_properties(class_name):
    """ Get a list of properties that are used by a particular Security class. """
    sec_class_handle = getattr(importlib.import_module('pyfintools.security.single'), class_name)
    base_class_handle = getattr(importlib.import_module('pyfintools.security.single'), 'Security')    
    base_props = constants.SECURITY_BASE_PROPERTIES
    additional_props = list(set(dir(sec_class_handle())) - set(dir(base_class_handle())))
    return base_props + additional_props

def get_security_class_name(cat_1_code, cat_2_code):
    class_name = None
    if 'CASH' == cat_1_code:
        if 'CASH' == cat_2_code:
            class_name = 'Cash'
    elif 'BD' == cat_1_code:
        if 'BDI' == cat_2_code:
            class_name = 'BondIndex'
        elif 'FIX' == cat_2_code:
            class_name = 'StraightBond'
        elif 'FRN' == cat_2_code:
            class_name = 'FloatingRateNote'
        elif 'OID' == cat_2_code:
            class_name = 'OriginalIssueDiscount'
    elif 'EQ' == cat_1_code:
        if 'EQI' == cat_2_code:
            class_name = 'EquityIndex'
        elif 'STOCK' == cat_2_code:
            class_name = 'CommonStock'
        elif 'PREF' == cat_2_code:
            class_name = 'PreferredStock'
        elif 'ETF' == cat_2_code:
            class_name = 'ETF'
        elif 'ETN' == cat_2_code:
            class_name = 'ETN'
    elif 'COM' == cat_1_code:
        if 'COMI' == cat_2_code:
            class_name = 'CommodityIndex'
        elif 'COMSPOT' == cat_2_code:
            class_name = 'CommoditySpot'
        else:
            raise NotImplementedError(f'Unknown commodity class: {class_name}.')
    elif 'RE' == cat_1_code:
        if 'REI' == cat_2_code:
            class_name = 'RealEstateIndex'
        else:
            raise NotImplementedError(f'Need to implement class for RE "{cat_2_code}".')
    elif 'HF' == cat_1_code:
        if 'HFI' == cat_2_code:
            class_name = 'HedgeFundIndex'
    elif 'IR' == cat_1_code:
        class_name = 'InterestRate'
    elif 'FX' == cat_1_code:
        if cat_2_code in ['FXFWD', 'FXSPOT', '']:
            class_name = 'FX'
        else:
            raise ValueError(f'Unsupported category information: {cat_1_code}-{cat_2_code}')
    elif 'FUT' == cat_1_code:
        if 'FUT' == cat_2_code:
            class_name = 'Future'
    elif 'ECON' == cat_1_code:
        if cat_2_code == 'PRC':
            class_name = 'EconomicPriceIndex'
    elif 'STRATEGY' == cat_1_code:
        class_name = 'Strategy'
    return class_name

def get_required_meta_data_fields(cat_1_code, cat_2_code):
    class_name = get_security_class_name(cat_1_code, cat_2_code)

def format_ccy_pair_and_tenor_info(target_ccy_pairs, target_tenors):
    """ Take arguments representing currency pairs and tenors, and output pairs of currency pair/tenor.
    
        Arguments:
             target_ccy_pairs: str/list - e.g., 'USD/JPY', 'AUD/EUR', etc.
             target_tenors: str/list - e.g. 'spot', '1m', '1y', etc.
        
        Returns:
            A list of tuples of currency pair / tenor
        
        Example: 
            target_ccy_pairs = ['USD/JPY', 'AUD/EUR']        
            target_tenors = ['1m', 'spot']
            Output: [('USD/JPY', '1m'), ('AUD/EUR', 'spot')]
    """
    
    if isinstance(target_ccy_pairs, str):
        if isinstance(target_tenors, str):
            # Case where both inputs are strings
            target_ccy_pairs = [target_ccy_pairs]
            target_tenors = [target_tenors.lower()]
        else:
            # If currency pair is a string, just repeat the ccy pair for each tenor
            target_ccy_pairs = [target_ccy_pairs for _ in target_tenors]
            target_tenors = [t.lower() for t in target_tenors]
    elif isinstance(target_tenors, str):
        # If tenor is a string, then repeat it for each currency pair
        target_ccy_pairs = target_ccy_pairs
        target_tenors = [target_tenors.lower() for _ in target_ccy_pairs]
    else:
        target_ccy_pairs = list(target_ccy_pairs)
        target_tenors = list(target_tenors)
        
        uniq_ccy_pairs = list(set(target_ccy_pairs))
        new_target_tenors = []
        new_target_ccy_pairs = []
        for tenor in set(target_tenors):
            new_target_tenors.extend([tenor.lower()] * len(uniq_ccy_pairs))
            new_target_ccy_pairs.extend(uniq_ccy_pairs.copy())
        target_tenors = new_target_tenors
        target_ccy_pairs = new_target_ccy_pairs
            
        if len(target_tenors) != len(target_ccy_pairs):
            raise ValueError('Dimensional mismatch between tenors and currency pairs.')

    # Perform additional data input consistency checks
    tenor_ccy_pairs = tuple(zip(target_ccy_pairs, target_tenors))
    if len(tenor_ccy_pairs) != len(set(tenor_ccy_pairs)):
        raise ValueError('There can be no duplicated currency pair/tenor tuple in the input arguments.')

    return target_ccy_pairs, target_tenors

def get_currency_pairs_from_inputs(long_currency, short_currency):
    """ Method to parse the long/short currency inputs (which can be either lists or strings),
          and return a list of currency pairs. """
    if isinstance(short_currency, str):
        if isinstance(long_currency, str):
            long_currency = [long_currency]
            short_currency = [short_currency]
        else:
            long_currency = list(long_currency)
            short_currency = [short_currency] * len(long_currency)
    else:
        short_currency = list(short_currency)
        if isinstance(long_currency, str):
            long_currency = [long_currency] * len(short_currency)
    currency_pairs = [ccy[0] + '/' + ccy[1] for ccy in zip(short_currency, long_currency)]
    return currency_pairs


class FXHelper(object):
    def __init__(self, base_currency, quote_currency, labels, tenor=None, cross_currency=None):
        if cross_currency is None:
            cross_currency = constants.DEFAULT_CROSS_CURRENCY
        self.base_currency = np.array(base_currency, dtype=str)
        self.quote_currency = np.array(quote_currency, dtype=str)
        self.labels = np.array(labels, dtype=str)
        if tenor is None:
            self.tenor = np.array([''] * self.labels.size)
        else:
            self.tenor = np.array(tenor)
        self.cross_currency = cross_currency

    def get_ccy_instructions(self, target_ccy_pairs, target_tenors=None):
        if target_tenors is None:
            target_tenors = np.array([''] * len(list(target_ccy_pairs)), dtype=str)
        else:
            target_tenors = np.array(target_tenors)

        assert len(target_ccy_pairs) == target_tenors.size, 'Dimensional mismatch in arguments.'
        indices = np.arange(len(target_ccy_pairs))

        uniq_tenors = set(target_tenors)
        instructions = [None] * len(target_tenors)
        for tenor in uniq_tenors:
            idx_tenor = indices[target_tenors == tenor]            
            for idx in idx_tenor:
                target_ccy_tpl = tuple(target_ccy_pairs[idx].split('/'))
                instructions[idx] = self._get_ccy_pair_locations(target_ccy_tpl, tenor)
        required_labels = self._get_required_labels(instructions)
        return instructions, required_labels
    
    def create_exchange_rates_from_instructions(self, instructions, ts_panel):
        ts_constr = []        
        for instr in instructions:
            if instr is None:
                ts = pd.Series(np.nan * np.ones_like(ts_panel.index, dtype=float), index=ts_panel.index)
                ts_constr.append(ts)
            elif instr == 1:
                ts = pd.Series(np.ones_like(ts_panel.index, dtype=float), index=ts_panel.index)
                ts_constr.append(ts)
            elif isinstance(instr, list) and len(instr) <=2:
                ts = 1
                for sub_instr in instr:
                    if sub_instr['sign'] == 1:
                        ts *= ts_panel[sub_instr['label']]
                    elif sub_instr['sign'] == -1:
                        ts /= ts_panel[sub_instr['label']]
                    else:
                        raise ValueError('Invalid sign')
                ts_constr.append(ts)
            else:
                raise ValueError('Unsupported instructions')
        return pd.concat(ts_constr, axis=1)

    def get_metadata_from_instructions(self, instructions, meta, target_ccy_pairs, target_tenors=None):
        if isinstance(target_ccy_pairs, str):
            target_ccy_pairs = np.array([target_ccy_pairs], dtype=str)
        else:
            target_ccy_pairs = np.array(target_ccy_pairs, dtype=str)

        if target_tenors is None:
            target_tenors = np.array([''] * target_ccy_pairs.size, dtype=str)
        else:
            target_tenors = np.array(target_tenors)
        
        new_meta_list = []
        for j, _instr in enumerate(instructions):            
            base_ccy, quote_ccy = target_ccy_pairs[j].split('/')
            ccy_pair = target_ccy_pairs[j]
            tenor = target_tenors[j]
            name = '{}_{}'.format(ccy_pair, tenor)
            cat_1_code = constants.CATEGORY_1_CODE_FX
            if tenor == constants.TENOR_SPOT:
                cat_2_code = constants.CATEGORY_2_CODE_FX_SPOT
            else:
                cat_2_code = constants.CATEGORY_2_CODE_FX_FORWARD                
            new_meta_dict = dict(base_currency=base_ccy, 
                                 quote_currency=quote_ccy, 
                                 currency_pair=ccy_pair,
                                 tenor=tenor,
                                 name=name,
                                 category_1_code=cat_1_code,
                                 category_2_code=cat_2_code)
            if _instr is None or _instr == 1:
                new_meta_dict['ts_type'] = pyfintools.tools.fts.TS_TYPE_LEVELS
                new_meta_list.append(pd.Series(new_meta_dict))
            else:
                cols = [x['label'] for x in _instr]
                assert len(cols) <= 2, 'There should not be more than 2 currency pairs needed to construct target pair.'
                tmp_meta = meta[cols]
                if tmp_meta.shape[1] == 2:
                    single_meta = tmp_meta.iloc[tmp_meta.values[:,0] == tmp_meta.values[:,1], 0].to_dict()
                else:
                    single_meta = tmp_meta.iloc[:,0].to_dict()
                single_meta.update(new_meta_dict)
                new_meta_list.append(pd.Series(single_meta))
        
        # Create a data frame of the new meta data
        new_meta_df = pd.concat(new_meta_list, axis=1)
        new_meta_df.columns = new_meta_df.loc['name']
        return new_meta_df
    
    def _get_required_labels(self, instructions):
        required_labels = set()
        for instr in instructions:
            if isinstance(instr, list):
                required_labels |= set([x['label'] for x in instr])
        return list(required_labels)
        
    def _get_ccy_pair_locations(self, ccy_pair, tenor=''):
        idx_tenor = self.tenor == tenor
        sub_labels = self.labels[idx_tenor]
        avail_ccy_tuples = list(zip(self.base_currency[idx_tenor], self.quote_currency[idx_tenor]))
        required_locations = None
        base, quote = ccy_pair
        if base == quote:
            # If the currency pair is the identity (e.g. USD/USD, EUR/EUR, etc.)
            required_locations = 1
        elif ccy_pair in avail_ccy_tuples:
            # If the currency pair already exists, then get the ticker
            idx = avail_ccy_tuples.index(ccy_pair)
            required_locations = [dict(label=sub_labels[idx], sign=1)]
        elif (quote, base) in avail_ccy_tuples:
            # If the inverse currency pair exists, then get the ticker (and then invert the time series later)
            idx = avail_ccy_tuples.index((quote, base))
            required_locations = [dict(label=sub_labels[idx], sign=-1)]
        else:
            # ...otherwise, we can try to construct the exchange rate from other exchange rates
            #     by going through a 'cross' rate.
            if self.cross_currency is None:
                raise ValueError('The target currency pair cannot be constructed without specifying a cross currency.')
            elif ((base, self.cross_currency) in avail_ccy_tuples or (self.cross_currency, base) in avail_ccy_tuples) and \
               ((quote, self.cross_currency) in avail_ccy_tuples or (self.cross_currency, quote) in avail_ccy_tuples):
                if (base, self.cross_currency) in avail_ccy_tuples:
                    idx = avail_ccy_tuples.index((base, self.cross_currency))
                    required_locations = [dict(label=sub_labels[idx], sign=1)]
                else:
                    idx = avail_ccy_tuples.index((self.cross_currency, base))
                    required_locations = [dict(label=sub_labels[idx], sign=-1)]

                if (quote, self.cross_currency) in avail_ccy_tuples:
                    idx = avail_ccy_tuples.index((quote, self.cross_currency))
                    required_locations.append(dict(label=sub_labels[idx], sign=-1))
                else:
                    idx = avail_ccy_tuples.index((self.cross_currency, quote))
                    required_locations.append(dict(label=sub_labels[idx], sign=1))
        return required_locations
    