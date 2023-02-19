""" Defines functions to help read from Excel workbooks.
"""

import pandas as pd
import re

TS_FORMAT_DATASTREAM = 'datastream'

def read_excel_workbook(filename, sheetnames=None, skip_sheets=None, header=0):
    if skip_sheets is None:
        skip_sheets = []

    xls = pd.ExcelFile(filename)
    
    if sheetnames is None:
        sheetnames = xls.sheet_names
    
    data = dict()
    for sheet_name in sheetnames:
        if sheet_name in skip_sheets:
            continue
        else:
            data[sheet_name] = pd.read_excel(xls, sheet_name, header=header)
    return data

def read_excel_ts(filename, fmt, skip_sheets=None, header=0, **kwargs):
    """ Read in and format time series from all sheets of an excel workbook. 
        Arguments:
            filename: the path and file name of the Excel workbook
            fmt: a string representing the supported format type (currently 'datastream'), or else a function handle
                that can format the time series data in a single sheet.
            skip_sheets: list containing names of any sheets that should not be checked for time series data
    """
    
    # Read in the raw Excel data
    raw_data = read_excel_workbook(filename, skip_sheets=skip_sheets, header=header)

    ts_data = []
    meta_data = pd.DataFrame()
    for sheet_name in raw_data.keys():
        df_raw = raw_data[sheet_name]

        # Extract the monthly time series and meta data
        if callable(fmt):
            # If the 'fmt' argument is actually a function, then use it to clean the data
            _ts, sub_meta_data = fmt(df_raw, **kwargs)
        else:
            _ts, sub_meta_data = process_raw_excel_data(df_raw, fmt, **kwargs)
        meta_data = pd.concat([meta_data, sub_meta_data], axis=1)
        ts_data.append(_ts)

    # Combine all of the time series together
    meta_data = meta_data.T
    df = pd.concat(ts_data, axis=1)

    # Drop any duplicated columns
    df = df.loc[:,~df.columns.duplicated()]
    return df, meta_data

def process_raw_excel_data(df_raw, fmt, **kwargs):
    if not df_raw.size:
        return df_raw, df_raw
    elif TS_FORMAT_DATASTREAM == fmt:
        return process_raw_excel_data_datastream(df_raw)
    else:
        raise ValueError(f'Unsupported Excel time series format type: {fmt}')

def process_raw_excel_data_datastream(df_raw):
    
    # Only keep columns that do not contain errors
    non_error_cols = [col for col in df_raw.columns if re.search('#ERROR', col) is None]
    df_raw = df_raw[non_error_cols]

    # Extract the dates
    date_index = _extract_date_index(df_raw.values[2:,0])
    ts_info = dict(name=list(df_raw.columns)[1:], 
                   ticker=list(df_raw.iloc[0].values)[1:],
                   denominated_currency=list(df_raw.iloc[1].values)[1:])

    ts_info['sec_code'] = [x.split('(')[0] for x in ts_info['ticker']]
    
    series_type_codes = []
    for tkr in ts_info['ticker']:
        if '(' in tkr:
            series_type_codes.append(tkr.split('(')[1].replace(')', ''))
        else:
            series_type_codes.append('')
    ts_info['series_type_code'] = series_type_codes
    values = df_raw.iloc[2:,1:].values
    df = pd.DataFrame(values, index=date_index, columns=ts_info['ticker'])

    meta_data = pd.DataFrame.from_dict(ts_info).T
    meta_data.columns = meta_data.loc['ticker']
    return df, meta_data

def _extract_date_index(raw_date_info):
    if isinstance(raw_date_info[0], str) and re.match('^Q[1-4]', raw_date_info[0]) is not None:
        # Data is in quarterly format (e.g. Q1 2012)
        qrtly_date_sfx = dict(Q1='-03-31', Q2='-06-30', Q3='-09-30', Q4='-12-31')
        q_dates = []
        for d in raw_date_info:
            d = d.replace(' ', '')
            q = d[:2]
            y = d[2:]
            q_dates.append(y + qrtly_date_sfx[q])
        date_index = pd.DatetimeIndex(q_dates)
    elif (isinstance(raw_date_info[0], str) and len(raw_date_info[0]) == 4) or \
          isinstance(raw_date_info[0], int) :
        # Annual date format (e.g. '2001', '1994', etc.)
        date_index = pd.DatetimeIndex([ f'{d}-12-31' for d in raw_date_info])
    else:
        # Assume dates are in normal format
        date_index = pd.DatetimeIndex(raw_date_info)
    return date_index

