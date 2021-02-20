import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import Line, Marker


def plot_ts_1(input_ts):
    """ Plot time series with an interactive plot.

        Arguments:
            input_ts: the time series (possibly multiple) to be plotted on the same y-axis.
    """
    # Make sure the input is a DataFrame
    input_ts = pd.DataFrame(input_ts)

    fig = make_subplots()
    
    # Add traces
    for col in input_ts.columns:
        tmp_ts = input_ts[col].dropna(axis=0)
        fig.add_trace(go.Scatter(x=tmp_ts.index, y=tmp_ts, name=col))
    
    # Set x-axis title
    fig.update_xaxes(title_text="Date")
    fig.show()
    
def plot_ts_2(ts, symbols_left, symbols_right, start=None, end=None, 
                                  labels_left=None, labels_right=None):
    """Plot time series from a Data Frame on both the left and right axes.
    
    Arguments:
        ts (DataFrame): the pandas DataFrame to be plotted
        symbols_left (list): symbols in 'ts' to be plotted on the left axis
        symbols_right (list): symbols in 'ts' to be plotted on the right axis
        start (datetime.date): the start date for the line plot (optional). 
                        If not specified, just use the first date from 'ts'
        end (datetime.date): the start date for the line plot (optional). 
                        If not specified, just use the first date from 'ts'
        labels_left (list): the labels to use for symbols_left on the plot
        labels_right (list): the labels to use for symbols_right on the plot        
    
    Returns: None
    """
    if labels_left is None:
        labels_left = symbols_left
    if labels_right is None:
        labels_right = symbols_right

    # Restrict to dates when at least some values are available
    # For any missing values, use the previous available value
    sub_ts = ts[symbols_left + symbols_right]
    sub_ts = sub_ts.dropna(axis=0, how='all')

    # Restrict analysis to between the start/end dates (if provided)
    if start is not None:
        if end is not None:
            sub_ts = sub_ts[start:end]
        else:
            sub_ts = sub_ts[start:]
    elif end is not None:
        sub_ts = sub_ts[:end]

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    for j, symbol in enumerate(symbols_left):
        tmp_ts = sub_ts.loc[:,[symbol]].dropna(axis=0)
        label = labels_left[j]
        fig.add_trace(
            go.Scatter(x=tmp_ts.index, y=tmp_ts[symbol], name=label),
            secondary_y=False,
        )

    for j, symbol in enumerate(symbols_right):
        tmp_ts = sub_ts.loc[:,[symbol]].dropna(axis=0)
        label = labels_right[j]
        fig.add_trace(
            go.Scatter(x=tmp_ts.index, y=tmp_ts[symbol], name=label),
            secondary_y=True,
        )

    # Add figure title
    fig.update_layout(
        title_text="{} vs. {}".format(', '.join(labels_left), 
                                      ', '.join(labels_right))
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Date")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>{}</b>".format(', '.join(labels_left)), secondary_y=False)
    fig.update_yaxes(title_text="<b>{}</b>".format(', '.join(labels_right)), secondary_y=True)

    fig.show()

def plot_candlestick(ts, open='open', high='high', low='low', 
                         close='close', time='date'):
    """Plot an interactive candlestick chart for time series OCHL data.
    Arguments:
    ts, open, high, low, close, time
    
    Returns: None
    """
    def find_column_index(ts, col):
        cols = [c.lower() for c in ts.columns]
        if col.lower() in cols:
            return cols.index(col.lower())
        else:
            return -1
    
    args = []
    # Add the column names, accounting for differences in case
    cs_cols = [open, high, low, close]
    for j, col in enumerate(cs_cols):
        col_idx = find_column_index(ts, col)
        if col_idx == -1:
            raise ValueError('Column not present in data frame: ' +\
                             '{}'.format(col))
        else:
            sub_ts = ts.iloc[:,col_idx]
            args.append(sub_ts)

    # Add the date/time information. 
    # Check if Date/time appears as a column in the DataFrame. 
    #   If not, then assume the index represents the date/time information.
    time_idx = find_column_index(ts, time)
    if time_idx == -1:
        if not isinstance(ts.index.values[0], np.datetime64):
            raise ValueError('Date/time info must be included as a column or else ' + \
                             'as an index with datetime entries.')
        ts.index.name = time
        ts = ts.reset_index()
        time_idx = find_column_index(ts, time)
    sub_ts = ts.iloc[:,time_idx]
    args.append(sub_ts)
        
    fig = plotly.figure_factory.create_candlestick(*args)
    plotly.offline.plot(fig)


def plot_candlestick_volume(ts, open='open', high='high', low='low', 
                         close='close', time='date', volume='volume'):
    """Plot an interactive candlestick chart for time series OHLC data.
    Arguments:
    ts, open, high, low, close, time, volume
    
    Returns: None
    """
    def find_column_index(ts, col):
        cols = [c.lower() for c in ts.columns]
        if col.lower() in cols:
            return cols.index(col.lower())
        else:
            return -1
    
    args = []
    # Add the column names, accounting for differences in case
    cs_cols = [open, high, low, close]
    for j, col in enumerate(cs_cols):
        col_idx = find_column_index(ts, col)
        if col_idx == -1:
            raise ValueError('Column not present in data frame: ' +\
                             '{}'.format(col))
        else:
            sub_ts = ts.iloc[:,col_idx]
            args.append(sub_ts)

    # Add the date/time information. 
    # Check if Date/time appears as a column in the DataFrame. 
    #   If not, then assume the index represents the date/time information.
    time_idx = find_column_index(ts, time)
    if time_idx == -1:
        if not isinstance(ts.index.values[0], np.datetime64):
            raise ValueError('Date/time info must be included as a column or else ' + \
                             'as an index with datetime entries.')
        ts.index.name = time
        ts = ts.reset_index()
        time_idx = find_column_index(ts, time)
    sub_ts = ts.iloc[:,time_idx]
    args.append(sub_ts)
        
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2
    )

    fig.add_trace( 
        go.Candlestick(
                        name = 'Candlestick',
                        x=args[-1],
                        open=args[0],
                        high=args[1],
                        low=args[2],
                        close=args[3]),
                  row=1, col=1)

    vol_idx = find_column_index(ts, volume)
    fig.add_trace(go.Bar(x=args[-1], y=ts.iloc[:,vol_idx]),
                  row=2, col=1)

    plotly.offline.plot(fig)    
