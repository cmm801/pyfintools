import findatadownload.download_ts as fdts
import findatadownload.download_meta as fdmeta


def download_time_series(data_source, base_ticker, start=None, end=None,
                         frequency=None, period=None, session=None):
    """ Download data for a single base ticker (which may yield one or more time series).

        Returns pandas DataFrame object with datetimes as the index and tickers as columns.
    """
    return fdts.download_time_series(data_source, base_ticker, start=start, end=end,
                         frequency=frequency, period=period, session=session)


def combine_symbol_lists(base_path):
    """ Combine symbol lists from EODDATA into a single DataFrame.
    """
    return fdmeta.combine_symbol_lists(base_path)
