import os
import datetime
import numpy as np
import pandas as pd
import yfinance as yf

EODDATA_EXCHANGES = ['NYSE', 'NASDAQ', 'AMEX']


def clean_fred_metadata(full_filename: str):
    """Extract meta data from the README file for a FRED data list.
    
    Arguments:
        full_filename (str): the full filename and path to the text file 
            returned from FRED, containing information on the time series.
            
    Returns:
        df (pd.DataFrame): A DataFrame object containing the meta data
    """
    data = []
    SERIES_ID = 'Series ID'
    with open(full_filename, "r") as f:

        def fast_forward_to_next_series():
            line = f.readline()
            while line and not line.startswith(SERIES_ID):
                line = f.readline()
            return line.replace('\n', '').strip()

        info = {SERIES_ID: ''}
        key = fast_forward_to_next_series()
        for line in f:
            line = line.replace('\n', '').strip()
            if line.startswith("-----"):
                next_line_is_val = True
            elif not line:
                key = None
                next_line_is_val = False
            elif line == SERIES_ID:
                data.append(info)
                key = SERIES_ID
                info = {SERIES_ID: ''}
            elif line.startswith('Notes'):            
                data.append(info)
                key = SERIES_ID
                info = {SERIES_ID: ''}
                fast_forward_to_next_series()            
            elif key is None:
                key = line
                info[key] = ''
            else:
                info[key] += line

        df = pd.DataFrame(data)
        df = df.set_index('Series ID', drop=False)
        return df

def _parse_eoddata_symbol_list(filename):
    """ Parse the meta data from EODDATA.com, and return a DataFrame. """

    with open(filename) as f:
        raw_data = f.read()

        # Separate the rows and columns
        parsed_data = [row.split('\t') for row in raw_data.split('\n')]

        # Drop the first row, which is the header
        parsed_data = parsed_data[1:]

        # Create a data frame
        symbols = pd.DataFrame(parsed_data, columns=['symbol', 'name']).dropna()
        return symbols
    
def _get_file_creation_time(filename):
    """ Get the time that the file was created.
    """
    if os.path.isfile(filename):
        mtime = os.path.getmtime(filename)
    else:
        mtime = 0

    last_modified_date = datetime.datetime.fromtimestamp(mtime)
    return last_modified_date
    
def combine_symbol_lists(base_path):
    """ Combine symbol lists from www.eoddata.com.
    """
    # Define a format string for the creation date
    fmt_str = '%Y-%m-%d'
    
    # Combine all symbol lists
    df = pd.DataFrame()
    for exchange in EODDATA_EXCHANGES:
        filename = f'{base_path}/{exchange}.txt'
        symbols = _parse_eoddata_symbol_list(filename)
        symbols['exchange'] = exchange
        
        # Get the time the file was created
        last_modified_date = _get_file_creation_time(filename)
        created_str = datetime.datetime.strftime(last_modified_date, fmt_str)
        symbols['valid_after'] = created_str

        # Combine the new symbols into the larger data frame
        df = pd.concat([df, symbols], axis=0)

    # Drop rows with an empty 'name' field
    idx_has_name = np.array([isinstance(x, str) and len(x) > 0 for x in df.name.values])
    df = df.loc[idx_has_name]

    # Drop duplicates
    df = df.drop_duplicates(['symbol', 'name'], keep='first')
    df = df.set_index('symbol')
    return df

