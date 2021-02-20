""" A set of classes that allow simultaneous access to both time series and meta data.

    The three main modules in this Security package are 'single', 'panel', and 'timeseries'. 
    Additionally, there is are 'constants' and 'helper' modules that support the three main modules.

    The modules 'single', 'panel', and 'timeseries' all have their own roles to play (as described
        below), but they are all structured along similar lines. Each of these three modules starts 
        by defining a base class (Security for 'single', Panel for 'panel', and TimeSeries for 
        'timeseries'). From these base classes, a number of subclasses are defined that allow for 
        different behavior/treatment. For example, all of these modules will have an AssetIndex class
        that inherits from the base class, and then an GenericIndex class that inherits from AssetIndex,
        and then an EquityIndex that inherits from GenericIndex. 
            
    The 'single' module has classes (inheriting from the Security class) that correspond to a single 
        type of time series data, and provides an interface that offers access to both the meta and 
        time series data. The 'panel' module has classes (inheriting from the Panel class) that are
        effectively a group one or more different types of time series, and again allows access
        to both meta and time series data.
        
    The 'single' and 'panel' objects are initialized by providing the meta data for the
        different securities (e.g. for equity indices, fx rates, etc.). 
        They also contain a variable, 'ts_db', that offers access to the time series data if desired.
        The meta data that is provided is in the form of two different pandas DataFrames:
            security_info: contains the key information about the different types of financial securities
            ticker_info: contains information about which ticker symbols are associated with data
                for this particular security object.
        
    While the 'timeseries' module is structured in a similar way as the 'single' and 'panel' 
        modules, the time series and meta data are both used to initialize TimeSeries classes, and
        therefore there is no need to provide any connection to a database.

    Note that when data is obtained from the 'single' or 'panel' objects using the 'get_time_series' method,
        the data is returned in a TimeSeries object form. This TimeSeries object then contains the 
        time series DataFrame in a variable called 'ts', and contains the meta data in another variable
        called 'meta'.
"""