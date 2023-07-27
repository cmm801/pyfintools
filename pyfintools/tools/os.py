""" Contains a few miscellaneous functions involving the operating system
"""

import os
import getpass


def is_non_zero_file(fpath):
    """ Returns False if a file does not exist or is empty; returns True otherwise, 
    """
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def get_user_name():
    """ Get the user name of the current user. 
        The current implementation may be platform dependent. If it raises an exception on another platform, then
            this function should be enhanced to work for the new platform (while not breaking current supported platforms)
    """
    return getpass.getuser()