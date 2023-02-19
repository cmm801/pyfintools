import os
#import pkg_resources
import numpy as np

########################################################################
# Path to External Data - this will be different for each user.
# TODO - find a better solution than hard-coding the path here.
DATA_PATH = '/Users/chris/OneDrive/programming/projects/finance/secdata'
########################################################################

# Tuple of numeric data types
NUMERIC_DATA_TYPES = (int, np.int64, float, np.float32, np.float64)

# Tuple of boolean data types
BOOLEAN_DATA_TYPES = (bool)

# Compounding frequencies are usually an integer number of times per year
# Rather than use 'infinity' (np.inf) to represent continuous compounding, we instead use -1
CONTINUOUS_COMPOUNDING = -1

# Optimization methods - we define these here instead of in the optimization
# package because they also get used by the SAA packages and we want to avoid cross-dependencies
METHOD_MEAN_VARIANCE = 'meanvar'
METHOD_MIN_VARIANCE = 'minvar'
METHOD_CVAR = 'cvar'
METHOD_ROBUST = 'robust'
METHOD_ROBUST_ERC = 'robust_erc'
