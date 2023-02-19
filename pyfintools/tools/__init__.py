""" This package contains a number of generic tools that can be applied to a wide range of problems.

    The modules appearing in this package are:
    
    optim: contains optimization methods, including mean-variance, robust (Ceria-Stubbs), 
        Equal Risk Contribution (ERC aka risk parity), and robust ERC (proprietary method).
        Also includes some code for working with constraints from various libraries (e.g. scipy and cvxpy).
    
    freq: helps to manipulate and extract time series frequency information, such as the number
        of periods/observations per year in a given time series, or extracting the frequency
        of a time series object.
    
    excel: helps read from Excel workbooks
    
    factormodel: contains a generic factor model class, as well as all of the required logic for 
        estimating betas and covariance matrices; simulating future or historical returns; 
        estimating the uncertainty covariance matrix (which can be used in robust optimization)
    
    fts: contains a library of common time series functions, as well as 
        providing an extension to pandas Series/DataFrame objects. To use the pandas extension, 
        you must first define a 'ts_type' variable on the pandas object and set it to one of 'levels',
        'returns', 'log_returns', 'rates' or 'dates'. Then, for supported time series types, you can
        convert between price 'levels' and 'returns and vice versa.
        This library also contains functions which can convert the data frequency.
        Some of the statistics included here are annual returns, cumulative returns, 
        volatility, CVaR, VaR, Sharpe ratio, etc.
            
    utils: contains a number of miscellaneous utility functions, which are used mainly by other modules.
        One useful function here is 'search', which allows you to search the entire code library 
        here for a given pattern. This makes it easier to see if, for example, a function that
        you want to change is used anywhere else in the code.
    
    bond: contains classes for calculating the prices of a single bond or portfolio of bonds.
    
    equity: contains classes for estimating equity prices (e.g. Implied ERP)
    
    os: contains a few miscellaneous functions involving the operating system
    
    sim: contains numerous functions for performing simulations. Provides functions for 
        simulating normal/lognormal data as well as moving and stationary block bootstraps.
    
    stats: contains a class for performing regressions, as well as code for
        unsmoothing auto-correlated time series (using the Geltner technique)
    
    yieldcurve: contains functions/classes for working with yield curves, and converting
        from spot to forward yields and vice versa

"""