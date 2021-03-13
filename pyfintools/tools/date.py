""" This module includes functions that help work with datetime/timestamp objects.
"""

import collections
import datetime
import pytz


def convert_timestamp_to_datetime(timestamp, to_tz=None, include_tz=True):
    """ Convert a UTC timestamp (or list/array of timestamps) into datetime object(s).
    
        Arguments:
            to_tz: the pytz timezone object in which we want the output date expressed.
            include_tz: (bool) whether or not to include the time zone info in the output.
            
        Returns:
            If the input is a single timestamp, then the output is that timestamp converted
                into a datetime object. 
            If the input is a list/array of timestamps, then the output is a list of those
                timestamps converted into datetime objects.
    """
    if to_tz is None:
        to_tz = pytz.utc

    if not isinstance(timestamp, collections.Iterable):
        input_values = [timestamp]
    else:
        input_values = timestamp
    
    output_values = []
    for idx, t in enumerate(input_values):
        d = datetime.datetime.utcfromtimestamp(t)
        d_utc = pytz.utc.localize(d)

        d_target = d_utc.astimezone(to_tz)

        if not include_tz:
            d_target = d_target.replace(tzinfo=None)
        
        output_values.append(d_target)
            
    if not isinstance(timestamp, collections.Iterable):
        return output_values[0]
    else:
        return output_values
