""" Packages that provide generic tools and specific models for CMAs and SAAs.

    The main sub-packages that appear in this package are:

    **tools**: this package contains a number of generic tools that can be \ 
        applied to a wide range of problems. Examples of the modules in \ 
        this package are:
            * optim: contains optimization methods
            * freq: helps manipulate and extract time series frequency information
            * excel: helps read from Excel workbooks
            * factormodel: contains a generic factor model class
            * fts: contains a library of common time series functions, as well as\ 
                  providing an extension to pandas Series/DataFrame objects
            * utils: contains a number of miscellaneous utility functions
            * bond: contains functions for calculating the prices of a single \ 
                  bond or portfolio of bonds
            * equity: contains models for estimating equity prices\ 
                  (e.g. Implied ERP)
            * os: contains a miscellaneous functions involving the operating system
            * sim: contains functions for performing simulations\ 
                  (e.g. Monte-Carlo and bootstrap)
            * stats: contains statistical functions such as regressions
            * yieldcurve: contains functions/classes for working with yield\ 
                  curves, and converting from spot to forward yields and vice versa


    **security**: this is a set of modules that provide simultaneous access to the \ 
        meta and time series data for financial securities. It is often important to \ 
        use the meta data (e.g. currency information, tenors, currency pairs for FX, \ 
        credit ratings, etc.) at the same time as the time series are being used. \ 
        Rather than have to keep track of all of this information in many different \ 
        objects, it is much easier to have all of the relevant data stored in the \ 
        same place.
      
    Other files:
    
    **constants**: defines constant parameters used throughout the codebase.
   
"""

    ###############################################################################
    ###########  This is the maximum length of a line - 79 characters  ############
    ###############################################################################
