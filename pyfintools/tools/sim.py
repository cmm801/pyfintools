""" Provides functions for simulating normal/lognormal data as well as moving and stationary block bootstraps.
"""


import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


LOGNORMAL = 'lognormal'
BOOTSTRAP_MOVING = 'moving_bootstrap'
BOOTSTRAP_STATIONARY = 'stationary_bootstrap'


def simulate_normal(size, mean, cov, rand_state=None):
    if isinstance(size, int):
        n_sims = size
        output_size = (size, mean.size)
    else:
        n_sims = np.product(size)
        output_size = size + (mean.size,)
    
    if rand_state is None:
        rand_state = _get_random_state(seed=None)

    sim_rtns_long = rand_state.multivariate_normal(mean.ravel(), cov, n_sims)
    return sim_rtns_long.reshape(output_size, order='C')

def simulate_lognormal(size, mean, cov, rand_state=None):
    sim_rtns = simulate_normal(size=size, mean=mean, cov=cov, rand_state=rand_state)
    return np.exp(sim_rtns) - 1

def simulate_block_bootstrap(size, time_series, block_size, demean=True, 
                             rand_state=None, sim_type=BOOTSTRAP_STATIONARY):
    n_sims = np.product(size)
    n_obs = time_series.shape[0]
    if rand_state is None:
        rand_state = _get_random_state(seed=None)

    if sim_type == BOOTSTRAP_MOVING:
        locs = _get_moving_block_bootstrap_locations(n_sims=n_sims, 
                                                     n_obs=n_obs,
                                                     block_size=block_size,
                                                     rand_state=rand_state)
    elif sim_type == BOOTSTRAP_STATIONARY:
        locs = _get_stationary_block_bootstrap_locations(n_sims=n_sims, 
                                                         n_obs=n_obs,
                                                         block_size=block_size,
                                                         rand_state=rand_state)
    else:
        raise ValueError(f'Unsupported bootstrap type: {sim_type}')
    
    # Demean if necessary
    if demean:
        ts = time_series - time_series.mean(axis=0)
    else:
        ts = time_series
        
    # Use the random blocks to sample the time series
    if isinstance(ts, (pd.Series, pd.DataFrame)):
        ts = ts.to_numpy()

    sampled_ts_long = ts[locs,:]

    # Reshape the output before returning
    output_size = size + (time_series.shape[1],)    
    return sampled_ts_long.reshape(output_size, order='F')


def _get_moving_block_bootstrap_locations(n_sims, n_obs, block_size, rand_state):
    """ Follows the algorithm of Künsch (1989) 
                "The Jackknife and the Bootstrap for General Stationary Observations".
        The algorithm bootstraps with random blocks, all of the same length, and stitches
        them together. It avoids selecting blocks that might extend past the end of the array.
        This algorithm produces simulated time series that are NOT stationary.
        """
    n_starts = n_sims // block_size + 1

    # Randomly select blocks of indices
    idx_start = rand_state.choice(n_obs - block_size, size=n_starts, replace=True)[np.newaxis,:]
    idx_panel = np.arange(block_size)[:,np.newaxis] + idx_start
    idx = idx_panel.T.ravel()

    # Only keep n_timestamps of the block locations
    return idx[:n_sims]

def _get_stationary_block_bootstrap_locations(n_sims, n_obs, block_size, rand_state):
    """ Follows the algorithm of Politis and Romano (1994) "The Stationary Bootstrap".
        The algorithm bootstraps with random blocks, with different lengths, and stitches
        them together. The block lengths are chosen from a geometric distribution. Any blocks
        that extend past the end of the array are wrapped around to the beginning.
        This algorithm produces simulated time series that ARE stationary.
        """
    # Get the geometric distribution parameter that produces average blocks of block_size
    p = 1 / block_size

    # Select the block sizes from geometric distribution, and ensure there are enough
    lengths = np.array([], dtype=int)
    while lengths.sum() < n_sims:
        n_blocks = (n_sims - int(lengths.sum())) // block_size + 1
        extra_lengths = rand_state.geometric(p, n_blocks)
        lengths = np.hstack([lengths, extra_lengths])

    # randomly select the start locations for the blocks
    start_locs = rand_state.choice(n_obs, size=lengths.size, replace=True)

    # Add new blocks to the list
    blocks = []
    for j in range(lengths.size):
        b = lengths[j]
        blocks.append(start_locs[j] + np.arange(b))

    # Combine all of the blocks together
    locs = np.hstack(blocks).ravel()

    # Only keep n_sims blocks
    locs = locs[:n_sims]

    # Have locations wrap around to the beginning
    locs = locs % n_obs
    return locs
        
def _get_random_state(seed):
    if seed is None:
        seed = np.random.randint(2 ** 31)
    return np.random.RandomState(seed)
    