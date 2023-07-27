import numpy as np

from enum import Enum


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
class OptimMethods(Enum):
    MEAN_VARIANCE = 'meanvar'
    MIN_VARIANCE = 'minvar'
    CVAR = 'cvar'
    ROBUST = 'robust'
    ERC = 'erc'          # Equal Risk Contribution (aka risk parity)
    RISK_PARITY = 'erc'  # This is an alias for ERC
    ROBUST_ERC = 'robust_erc'
