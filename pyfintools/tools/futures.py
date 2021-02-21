""" Contains functions useful for working with Futures data.
"""
import datetime
import numpy as np
import os
import pandas as pd

import secdb.constants


CBOE_VIX_FUTURES_FIRST_DATE = '2013-01-01'  # First date that CBOE has historical futures data


def get_security_meta_data(series_type):
    """ Get a DataFrame containing security meta data.
    
        Arguments:
            series_type: (str) the code used for the .csv file containing the data.
        """
    filename = os.path.join(secdb.constants.DATA_PATH, \
                        f'meta_data/sec_info/{series_type.upper()}.csv')
    df = pd.read_csv(filename)
    return df

def get_futures_expiries(contract_code, start=None, end=None):
    """ Get all expiry dates between the start/end for a given futures contract code.
    """
    supported = ('VX')
    if contract_code not in supported:
        raise NotImplementedError('Only implemented for these futures: {}'.format(supported))

    if start is None:
        start = pd.Timestamp(CBOE_VIX_FUTURES_FIRST_DATE)
    else:
        start = pd.Timestamp(start)

    if end is None:
        end = pd.Timestamp.now() + pd.DateOffset(years=1)
    else:
        end = pd.Timestamp(end)

    # Get the expiry dates from the meta data
    meta_df = get_security_meta_data('FUT')
    
    # Only retrieve expiries that are between start/end
    qry_str = f'contract_code == "{contract_code}"'
    expiries = meta_df.query(qry_str).last_trade_date.astype(np.datetime64)
    expiries = [pd.Timestamp(d) for d in expiries if start <= d and d <= end]    
    return expiries
