""" Contains a number of miscellaneous utility functions, which are used mainly by other modules.

    Some useful functions: 
        'search': allows user to search the entire code library for a given pattern. 
            This makes it easier to see if, for example, a function that
            the user want to change is used anywhere else in the code.
"""

from pathlib import Path
import os
import yaml
import string
import numpy as np

import pyfintools.constants


def get_project_root() -> Path:
    """Returns project root folder."""
    return str(Path(__file__).parent.parent.parent)

def get_code_root():
    root_path = get_project_root()
    return os.path.join(root_path, 'pyfintools')

def get_test_root():
    root_path = get_project_root()
    return os.path.join(root_path, 'unittests')

def _get_config_path():
    code_path = get_code_root()
    return os.path.join(code_path, 'configs')
        
def _get_config_filepath_public():
    config_path = _get_config_path()
    return os.path.join(config_path, 'config.yml')

def _get_config_filepath_private():
    config_path = _get_config_path()
    return os.path.join(config_path, 'config_private.yml')

def read_config_public():
    conf_path = _get_config_filepath_public()
    with open(conf_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def read_config_private():
    conf_path = _get_config_filepath_private()
    with open(conf_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def search(search_string, extension='.py', verbose=1):
    """ Search for a string in the module, and print line numbers where it is found. 
        
        Arguments:
            extension: string/list/tuple - the file extensions that will be included in the results (e.g. '.py')
                       If no argument is provided, then all extensions will be included in the output.
    """
    if isinstance(extension, str):
        allowed_extensions = tuple([extension])
    else:
        allowed_extensions = tuple(extension)        

    output = []        
    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            _, ext = os.path.splitext(filename)
            if ext in allowed_extensions:
                with open(filename) as f:
                    details = dict()
                    for j, line in enumerate(f):
                        if search_string in line:
                            details[j+1] = line
                    if details:
                        output.append(filename)                        
                        if verbose == 1:
                            for n, txt in details.items():
                                output.append(f'\t{n}: {txt[:-1]}')
    [print(x) for x in output]

def is_boolean(x):
    return isinstance(x, pyfintools.constants.BOOLEAN_DATA_TYPES)

def is_numeric(x):
    return isinstance(x, pyfintools.constants.NUMERIC_DATA_TYPES)

def is_integer(x):
    if not is_numeric(x):
        return False
    else:
        return isinstance(x, (int, np.int64)) or x.is_integer()

def generate_random_string(n):
    return ''.join(np.random.choice(list(string.ascii_letters), n))

def find_weights_to_match_target(values, target_value):
    """ Find the weights on a set of values that will result in a weighted value that matches the target. 
        For example, 'values' could be durations for a set of bond indices, and 'target_value' could be 
            the target duration for an allocation. The resulting 'weights' would indicate how much the 
            individual bond indices should be weighted to produce the target duration.
    """
    # Make sure the values are sorted, but save the sort index so we put them back into the original order
    values = np.array(values)
    idx_sort = np.argsort(values)
    values_srt = values[idx_sort]
    
    # Initialize the weight vector
    weights_srt = np.zeros((values_srt.size,), dtype=float)
    idx_L = 0
    idx_U = 1

    while idx_U < values_srt.size and values_srt[idx_U] <= target_value:
        idx_L += 1
        idx_U += 1

    dur_L = values_srt[idx_L]
    dur_U = values_srt[idx_U]
    assert dur_L <= target_value and target_value <= dur_U, 'Available values do not bound target.'

    weights_srt[idx_L] = (dur_U - target_value) / (dur_U - dur_L)
    weights_srt[idx_U] = (target_value - dur_L) / (dur_U - dur_L)

    # Put the weights back into the original order
    idx_unsort = np.argsort(idx_sort)
    values = values_srt[idx_unsort]
    weights = weights_srt[idx_unsort]
    
    assert np.isclose(target_value, weights @ np.array(values)), 'Weights must sum to 1'
    return weights

