""" Defines a generic factor model class.

    Contains required logic for estimating betas and covariance matrices; 
    simulating future or historical returns; 
    estimating the uncertainty covariance matrix (which can be used in robust optimization)
"""

import numpy as np
import pandas as pd
import sklearn.linear_model

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import pyfintools.tools.fts
import pyfintools.tools.sim


# Specify default simulation methods for factors, idiosyncratic noise, and the risk-free rate
DEFAULT_FACTOR_SIM_TYPE = pyfintools.tools.sim.BOOTSTRAP_STATIONARY
DEFAULT_IDIO_SIM_TYPE = pyfintools.tools.sim.LOGNORMAL
DEFAULT_RFR_SIM_TYPE = None

# For parametric simulations, this is the default volatility for the risk-free rate
DEFAULT_RFR_SIM_VOLATILITY = 0.0025

# For block bootstrap simulations, define default parameters
DEFAULT_SIM_BLOCK_SIZE = 8


class BaseFactorModel(object):
    def __init__(self, rfr=np.nan, factor_exp_rtns=None, factor_cov=None, factor_names=None,
                         betas=None, idio_vols=None, asset_names=None):
        super(BaseFactorModel, self).__init__()
        self._initialize_instance_variables()
        
        # Set the risk free rate
        self.risk_free_rate = rfr
        
        # Set factor information
        self.factor_expected_rtns = factor_exp_rtns        
        self.factor_cov = factor_cov
        self.factor_names = factor_names

        # Set asset information
        self.asset_betas = betas
        self.asset_idio_vols = idio_vols
        self.asset_names = asset_names        

    def _initialize_instance_variables(self):
        # Initialize risk free rate variables
        self.risk_free_rate = np.nan
        
        # Initialize factor information
        self.factor_names = None        
        self.factor_expected_rtns = None
        self.factor_cov = None

        # Initialize asset information
        self.asset_names = None        
        self.asset_betas = None
        self.asset_idio_vols = None

    @property
    def n_factors(self):
        if self.factor_expected_rtns is not None:
            return self.factor_expected_rtns.size
        elif self.factor_cov is not None:
            return self.factor_cov.shape[0]
        else:
            return 0
    
    @property
    def factor_expected_rtns(self):
        return self._factor_expected_rtns
    
    @factor_expected_rtns.setter
    def factor_expected_rtns(self, rtns):
        if rtns is None:
            self._factor_expected_rtns = None
        else:
            rtn_mtx = np.array(rtns).ravel()
            if self.n_factors and self.n_factors != rtn_mtx.size:
                raise ValueError('Mismatch in number of implied factors.')
            else:
                self._factor_expected_rtns = rtn_mtx
    
    @property
    def factor_cov(self):
        return self._factor_cov
    
    @factor_cov.setter
    def factor_cov(self, cov):
        if cov is None:
            self._factor_cov = None
        else:
            cov_mtx = np.array(cov)
            if self.n_factors and self.n_factors != cov_mtx.shape[0]:
                raise ValueError('Mismatch in number of implied factors.')
            else:            
                self._factor_cov = np.array(cov_mtx)
    
    @property
    def factor_names(self):
        return self._factor_names
    
    @factor_names.setter
    def factor_names(self, names):
        if names is None:
            self._factor_names = None
        else:
            names = list(names)
            if self.n_factors and self.n_factors != len(names):
                raise ValueError('Mismatch in number of implied factors.')
            else:
                self._factor_names = names
    
    @property
    def n_assets(self):
        if self.asset_idio_vols is not None:
            return self.asset_idio_vols.size        
        elif self.asset_betas is not None:
            return self.asset_betas.shape[0]
        else:
            return None

    @property
    def asset_betas(self):
        return self._asset_betas
    
    @asset_betas.setter
    def asset_betas(self, betas):
        if betas is None:
            self._asset_betas = None
        elif not self.n_factors:
            raise ValueError('Betas must be be set after the factor covariance and/or returns.')
        else:
            beta_mtx = np.array(betas)
            if beta_mtx.shape[1] != self.n_factors \
                or (self.n_assets and beta_mtx.shape[0] != self.n_assets):
                raise ValueError('Betas must have dimensions (n_assets, n_factors).')
            elif self.n_assets and self.n_assets != beta_mtx.shape[0]:
                raise ValueError('Mismatch in number of implied factors.')
            else:
                self._asset_betas = beta_mtx
    
    @property
    def asset_idio_vols(self):
        return self._asset_idio_vols
    
    @asset_idio_vols.setter
    def asset_idio_vols(self, vols):
        if vols is None:
            self._asset_idio_vols = None
        else:
            vol_vec = np.array(vols).ravel()
            if self.n_assets and self.n_assets != vol_vec.size:
                raise ValueError('Mismatch in number of implied factors.')
            else:
                self._asset_idio_vols = vol_vec
    
    @property
    def asset_names(self):
        return self._asset_names
    
    @asset_names.setter
    def asset_names(self, names):
        if names is None:
            self._asset_names = None
        elif self.n_assets and self.n_assets != len(names):
            raise ValueError('Mismatch in number of implied factors.')
        else:
            self._asset_names = names
    
    @property
    def asset_cov(self):
        return self._asset_cov_systematic + self._asset_cov_idiosyncratic
    
    @property
    def _asset_cov_systematic(self):
        return self.asset_betas @ self.factor_cov @ self.asset_betas.T
    
    @property
    def _asset_cov_idiosyncratic(self):
        return np.diag(self.asset_idio_vols ** 2)    
    
    @property
    def asset_vols(self):
        return np.sqrt(np.diag(self.asset_cov))
    
    @property
    def factor_vols(self):
        return np.sqrt(np.diag(self.factor_cov))
    
    @property
    def asset_corr(self):
        vols = self.asset_vols + 1e-10 # Add a small number to prevent division by zero
        cov = self.asset_cov
        return np.diag(1/vols) @ cov @ np.diag(1/vols)
    
    @property
    def factor_corr(self):
        vols = self.factor_vols + 1e-10 # Add a small number to prevent division by zero
        cov = self.factor_cov
        return np.diag(1/vols) @ cov @ np.diag(1/vols)
    
    @property
    def asset_exp_exc_rtns(self):
        return self.factor_expected_rtns @ self.asset_betas.T
    
    @property
    def asset_exp_total_returns(self):
        return self.risk_free_rate + self.asset_exp_exc_rtns

    def plot_betas(self):
        fig, axes = plt.subplots(4, 3, figsize=(18, 25))
        fig.suptitle('Asset Factor Exposures', size=16, y=0.9)
        axes = axes.ravel()

        y_min = min(-0.1, self.asset_betas.min() * 1.2)
        y_max = self.asset_betas.max() * 1.2
        
        for j, name in enumerate(self.asset_names):
            sns.barplot(self.factor_names, self.asset_betas[j,:], palette="deep", ax=axes[j])
            axes[j].set_title(name)
            axes[j].set_ylim(y_min, y_max)
            axes[j].set_ylabel("Beta")


class FactorModel(BaseFactorModel):            
    def _initialize_instance_variables(self):
        super(FactorModel, self)._initialize_instance_variables()
        # Initialize time series
        self.risk_free_rtns_ts =  pd.Series([], dtype=float)
        self.factor_rtns_ts = pd.DataFrame()

        # Initialize default simulation types
        self.risk_free_rate_vol = DEFAULT_RFR_SIM_VOLATILITY        
        self.factor_sim_type = DEFAULT_FACTOR_SIM_TYPE
        self.idio_sim_type = DEFAULT_IDIO_SIM_TYPE
        self.rfr_sim_type = DEFAULT_RFR_SIM_TYPE
        
        # Initialize variables used in simulations
        self.sim_block_size = DEFAULT_SIM_BLOCK_SIZE
        self._seed = None
        self._rand_state = None

    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, sd):
        if sd is None:
            self._seed = np.random.randint(2 ** 31)
        else:
            self._seed = sd
        self._rand_state = np.random.RandomState(self.seed)

    def simulate_factors(self, size, freq=None, seed=None):
        n_steps, n_paths = _extract_size_info(size)
        
        # Get the number of periods per year so we can annualize returns/covariance        
        n_periods_per_year, freq = _get_n_periods_per_year(freq, self.risk_free_rtns_ts)
        
        # Set the random seed
        self.seed = seed        
        
        return self._sim_factors(n_steps, n_paths, n_periods_per_year=n_periods_per_year)
        
    def simulate_assets(self, size, freq=None, seed=None, demean=False):
        """ Simulate the assets using the factor model.
        
            Arguments:
                size: (tuple) a 2d tuple (n_obs, n_sims) where n_obs is how many observations per simulation,
                        and n_sims is how many separate simulations to perform.
                freq: (str) the frequency to be used for simulating. This argument can only be provided if the
                        return information stored in the FactorModel is a numpy array - otherwise, the simulation
                        will just use the same frequency as the pandas time series objects
                seed: (int) the random seed, which can be set for reproducible results
                demean: (bool) True/False - whether to demean the asset returns. Default setting is 'False'

            Output:
                Returns a 3d numpy array with dimensions (n_obs, n_sims, n_assets)
        """
        n_steps, n_paths = _extract_size_info(size)
        
        # Check the input time series for consistency
        _check_time_series_consistency(self.risk_free_rtns_ts, self.factor_rtns_ts)

        # Simulate the factor returns
        sim_factor_rtns = self.simulate_factors(size=(n_steps, n_paths), freq=freq, seed=seed)

        # Get the number of periods per year so we can annualize returns/covariance        
        n_periods_per_year, _ = _get_n_periods_per_year(freq, self.risk_free_rtns_ts)
        
        # Get the simulated systematic asset excess returns
        sim_asset_exc_rtns_sys = sim_factor_rtns @ self.asset_betas.T

        # Simulate idiosyncratic asset noise
        sim_asset_rtns_idio = self._sim_asset_idio_noise(n_steps, n_paths, n_periods_per_year=n_periods_per_year)
        
        # Simulate the risk-free rate returns
        sim_rfr_rtns = self._sim_asset_risk_free_rate(n_steps, n_paths, n_periods_per_year=n_periods_per_year)
        
        # Combine the pieces to get the simulated asset returns
        sim_asset_rtns = sim_rfr_rtns + sim_asset_exc_rtns_sys + sim_asset_rtns_idio

        # Demean the returns if requested
        if demean:
            _mean = sim_asset_rtns.mean(axis=(0, 1))
            return sim_asset_rtns - _mean
        else:
            return sim_asset_rtns
    
    def _sim_factors(self, n_steps, n_paths, n_periods_per_year):
        sim_params = dict(n_steps=n_steps, n_paths=n_paths, n_periods_per_year=n_periods_per_year)
        if pyfintools.tools.sim.LOGNORMAL == self.factor_sim_type:
            return self._sim_factors_lognormal(**sim_params)
        elif self.factor_sim_type in [pyfintools.tools.sim.BOOTSTRAP_MOVING, pyfintools.tools.sim.BOOTSTRAP_STATIONARY]:
            return self._sim_factors_block_bootstrap(**sim_params)
        else:
            raise ValueError('Unsupported simulation method for factors: {}'.format(self.factor_sim_type))

    def _sim_factors_lognormal(self, n_steps, n_paths, n_periods_per_year):
        prd_fac_rtns = self.factor_exp_exc_rtns / n_periods_per_year
        prd_fac_cov = self.factor_cov / n_periods_per_year
        return pyfintools.tools.sim.simulate_lognormal((n_steps, n_paths), mean=prd_fac_rtns,
                                               cov=prd_fac_cov, rand_state=self._rand_state)
    
    def _sim_factors_block_bootstrap(self, n_steps, n_paths, n_periods_per_year, sim_type=None):
        assert isinstance(self.factor_rtns_ts, (pd.DataFrame, pd.Series)), \
                             'The factor time series are not defined, and are required for bootstrap simulations.'
        sim_rtns = pyfintools.tools.sim.simulate_block_bootstrap((n_steps, n_paths), 
                                                            time_series=self.factor_rtns_ts, 
                                                            block_size=self.sim_block_size,
                                                            demean=True,
                                                            rand_state=self._rand_state, 
                                                            sim_type=pyfintools.tools.sim.BOOTSTRAP_STATIONARY)
        return sim_rtns + self.factor_expected_rtns / n_periods_per_year
    
    def _sim_factors_stationary_block_bootstrap(self, n_steps, n_paths, n_periods_per_year):
        assert isinstance(self.factor_rtns_ts, (pd.DataFrame, pd.Series)), \
                             'The factor time series are not defined, and are required for bootstrap simulations.'
        sim_rtns = pyfintools.tools.sim.simulate_block_bootstrap((n_steps, n_paths), 
                                                            time_series=self.factor_rtns_ts, 
                                                            block_size=self.sim_block_size,
                                                            demean=True,
                                                            rand_state=self._rand_state, 
                                                            sim_type=pyfintools.tools.sim.BOOTSTRAP_STATIONARY)
        return sim_rtns + self.factor_expected_rtns / n_periods_per_year        
    
    def _sim_asset_idio_noise(self, n_steps, n_paths, n_periods_per_year):
        sim_params = dict(n_steps=n_steps, n_paths=n_paths, n_periods_per_year=n_periods_per_year)
        if pyfintools.tools.sim.LOGNORMAL == self.idio_sim_type:
            return self._sim_asset_idio_noise_lognormal(**sim_params)
        else:
            raise ValueError('Unsupported simulation method for asset idiosyncratic noise: {}'.format(self.idio_sim_type))
    
    def _sim_asset_idio_noise_lognormal(self, n_steps, n_paths, n_periods_per_year):
        prd_idio_rtns = np.zeros_like(self.asset_idio_vols, dtype=float)
        prd_idio_cov = np.diag(self.asset_idio_vols ** 2) / n_periods_per_year
        return pyfintools.tools.sim.simulate_lognormal(size=(n_steps, n_paths), 
                                                  mean=prd_idio_rtns,
                                                  cov=prd_idio_cov, 
                                                  rand_state=self._rand_state)
    
    def _sim_asset_risk_free_rate(self, n_steps, n_paths, n_periods_per_year):
        sim_params = dict(n_steps=n_steps, n_paths=n_paths, n_periods_per_year=n_periods_per_year)
        if pyfintools.tools.sim.LOGNORMAL == self.idio_sim_type:
            return self._sim_asset_risk_free_rate_lognormal(**sim_params)
        else:
            raise ValueError('Unsupported simulation method for risk-free rates: {}'.format(self.rfr_sim_type))
    
    def _sim_asset_risk_free_rate_lognormal(self, n_steps, n_paths, n_periods_per_year):
        prd_rfr_mean = np.array([self.risk_free_rate / n_periods_per_year])
        prd_rfr_cov = np.array([[self.risk_free_rate_vol ** 2 / n_periods_per_year]])
        return pyfintools.tools.sim.simulate_lognormal(size=(n_steps, n_paths), 
                                                  mean=prd_rfr_mean,
                                                  cov=prd_rfr_cov, 
                                                  rand_state=self._rand_state)
        

        
# Factory method
def from_time_series(rfr, factor, asset, freq=None, ts_type='returns'):
    
    _check_time_series_consistency(rfr, factor, asset)

    # Align the frequencies of the time series
    rfr, factor, asset = _align_time_series(rfr, factor, asset, ts_type=ts_type)
        
    if ts_type == 'price_levels':
        raise NotImplementedError('Need to convert price levels to returns')
    
    # Initialize the factor model object
    fmodel = FactorModel()
    fmodel.risk_free_rtns_ts = rfr
    fmodel.factor_rtns_ts = factor

    if isinstance(fmodel.risk_free_rtns_ts, (pd.Series, pd.DataFrame)):
        fmodel.factor_names = fmodel.factor_rtns_ts.columns
        fmodel.asset_names = asset.columns

    # Get the number of periods per year so we can annualize returns/covariance
    n_periods_per_year, freq = _get_n_periods_per_year(freq, fmodel.risk_free_rtns_ts)
    
    # Calculate factor covariance
    fmodel.factor_cov = fmodel.factor_rtns_ts.cov() * n_periods_per_year
    
    # Make assumptions about future returns (these can be manually changed)
    fmodel.risk_free_rate = fmodel.risk_free_rtns_ts.mean() * n_periods_per_year
    fmodel.factor_expected_rtns = fmodel.factor_rtns_ts.mean() * n_periods_per_year    
    
    # Calculate the asset betas
    asset_exc_rtns = _get_asset_excess_returns(asset_rtns=asset, rfr_rtns=fmodel.risk_free_rtns_ts)
    betas, idio_std = estimate_asset_betas(factor_rtns=fmodel.factor_rtns_ts,
                                           asset_exc_rtns=asset)
    fmodel.asset_betas = betas
    fmodel.asset_idio_vols = idio_std * np.sqrt(n_periods_per_year)
    return fmodel

def estimate_asset_betas(factor_rtns, asset_exc_rtns, adj_for_acor=False, data_frequency=None, alpha=0.0):
    """ Run a multi-variate regression of the asset excess returns on the factor returns.
    
        Arguments:
            factor_rtns: pandas DataFrame with factor (excess) returns
            asset_exc_rtns: pandas Series or DataFrame with asset excess returns
            adj_for_acor: (bool) whether to adjust the regression for autocorrelated asset returns. Default is False.
            data_frequency: (str) the frequency of the input time series data. This is only required when the inputs
                for factor returns and asset returns are numpy arrays. If these are instead pandas objects, the
                data frequency will be inferred.
            alpha: (float) the normalization coefficient in a ridge regression. If 0.0 (the default value), no 
                normalization is performed.
    """
    if isinstance(asset_exc_rtns, pd.Series):
        asset_exc_rtns = pd.DataFrame(asset_exc_rtns)

    betas = []
    idio_std = []
    if isinstance(factor_rtns, (pd.Series, pd.DataFrame)):
        data_frequency = pyfintools.tools.freq.infer_freq(asset_exc_rtns.index, allow_missing=True)        
        factor_rtns_df, asset_exc_rtns_df = factor_rtns.align(asset_exc_rtns, axis=0, join='inner')

        factor_rtns = factor_rtns_df.to_numpy()
        asset_exc_rtns = asset_exc_rtns_df.to_numpy()
    elif data_frequency is None:
        raise ValueError('"data_frequency" must be specified when the factor/asset returns are numpy arrays.')
    
    # Get the rescaling factor for the idiosyncratic volatility
    periods_per_year = pyfintools.tools.freq.get_periods_per_year(data_frequency)

    # Loop through all of the asset time series and calculate the betas
    for j in range(asset_exc_rtns.shape[1]):
        asset_ER_j = asset_exc_rtns[:,j]
        reg_model = sklearn.linear_model.Ridge(alpha=alpha)
        idx_asset_rtns_are_good = ~np.isnan(asset_ER_j)
        idx_factor_rtns_are_good = ~np.any(np.isnan(factor_rtns), axis=1)
        idx = idx_factor_rtns_are_good & idx_asset_rtns_are_good
        if not np.any(idx):
            # If there are no good data points, then set the betas to NaN
            b = np.nan * np.ones((factor_rtns.shape[0],), dtype=float)
            s = np.nan
        else:
            # Get the betas from a linear regression
            res = reg_model.fit(X=factor_rtns[idx,:], y=asset_ER_j[idx])
            b = res.coef_
            
            # Get the idiosyncratic standard deviation
            yhat = res.predict(factor_rtns[idx,:])
            residual = asset_ER_j[idx] - yhat
            s = residual.std() * np.sqrt(periods_per_year)

        # Save the beta and idio std to the list
        betas.append(b)
        idio_std.append(s)

    # Make sure the betas are in a numpy array
    asset_betas = np.array(betas, dtype=float)
    
    # Adjust betas if desired, and then return the result
    if adj_for_acor:
        adj_betas = adjust_betas_for_autocorrelation(factor_rtns, asset_exc_rtns, asset_betas, 
                                                     data_frequency=data_frequency)

        # The idiosyncratic volatility is not well defined if we adjust for auto-correlation, so we don't return it
        return adj_betas, None
    else:
        return asset_betas, np.array(idio_std, dtype=float)


def adjust_betas_for_autocorrelation(factor_rtns, asset_exc_rtns, asset_betas, data_frequency=None):
    """ Adjust asset betas for autocorrelation by unsmoothing the residual of the unadjusted regression. 
        Instead of unsmoothing all time series at the start, we first give the factors a chance to
            explain the assets' autocorrelation. This is because some of the factors themselves have highly
            autocorrelated time series (like high yield bonds and commodities), and unsmoothing the asset
            returns from the beginning would distort their relationship with the factors.
            
        Therefore, this approach works in several steps:
            1) Find the residual of the original regression, and unsmooth it
            2) Perform a multivariate regression of the unsmoothed residual returns on the factor returns.
            3) Return the sum of the original betas with the betas from the unsmoothed residual.
    """
    # 0) Convert the factor and asset returns to numpy arrays
    factor_rtns, asset_exc_rtns = _format_factor_and_asset_returns(factor_rtns, asset_exc_rtns)

    # 1) Find the residual of the regression and unsmooth it
    asset_exc_rtns_hat = factor_rtns @ asset_betas.T    
    residual = asset_exc_rtns - asset_exc_rtns_hat
    unsmth_resid = pyfintools.tools.stats.geltner_unsmooth(residual, skipna=True)

    # 2) Calculate the asset betas of the unsmoothed residuals
    resid_asset_betas, _ = estimate_asset_betas(factor_rtns, unsmth_resid, data_frequency=data_frequency)
    
    # 3) Return the sum of the original betas with the betas from the unsmoothed residual.
    return asset_betas + resid_asset_betas

def _format_factor_and_asset_returns(factor_rtns, asset_rtns):
    if isinstance(factor_rtns, (pd.Series, pd.DataFrame)):
        factor_rtns, asset_rtns = factor_rtns.align(asset_rtns, axis=0, join='inner')
        factor_panel = factor_rtns.to_numpy()
        asset_vals = asset_rtns.to_numpy()
    else:
        factor_panel, asset_vals = factor_rtns.copy(), asset_rtns.copy()

    # Make sure there are no NaN values in the factors
    idx_nan = np.any(np.isnan(factor_panel), axis=1)
    factor_panel = factor_panel[~idx_nan,:]
    asset_vals = asset_vals[~idx_nan,:] 
    return factor_panel, asset_vals
    
def estimate_return_uncertainties(factor_rtns, asset_exc_rtns, n_sims=100, seed=None):
    if seed is None:
        seed = np.random.randint(2 ** 31)
    rand_state = np.random.RandomState(seed)

    # Align the dates of the return series
    factor_panel, asset_vals = _format_factor_and_asset_returns(factor_rtns, asset_exc_rtns)
    
    # Get the data frequency of the factor and asset time series
    data_frequency = pyfintools.tools.freq.infer_freq(asset_exc_rtns.index, allow_missing=True)
        
    # Align the columns so the series with the most history are on the left
    asset_panel, idx_revert = _order_columns_by_first_nan(asset_vals)
    first_rows = _find_first_non_nan_rows(asset_panel)
    n_rows, n_cols = asset_panel.shape   
    bootstrap_rtns = []
    bootstrap_betas_mean = np.zeros((n_cols, factor_panel.shape[1]), dtype=float)
    bootstrap_betas_sum_of_sq = np.zeros((n_cols, factor_panel.shape[1]), dtype=float)
    for _ in range(n_sims):
        bootstrap_locs = _generate_bootstrap_locations_for_regression_resampling(rand_state, n_rows, n_cols, first_rows)

        # Get the bootstrapped asset and factor returns
        col_idx = np.tile(np.arange(n_cols).reshape(-1, 1), n_rows)
        btrp_asset_rtns = asset_panel[bootstrap_locs, col_idx.T]
        btrp_asset_rtns[bootstrap_locs == -1] = np.nan

        btrp_factor_rtns = factor_panel[bootstrap_locs[:,0], :]
        betas, idio_std = estimate_asset_betas(btrp_factor_rtns, btrp_asset_rtns, data_frequency=data_frequency)

        mu = (btrp_factor_rtns @ betas.T).mean(axis=0)
        bootstrap_rtns.append(mu)
        bootstrap_betas_mean += 1/n_sims * betas
        bootstrap_betas_sum_of_sq += 1/n_sims * (betas ** 2)

    # Calculate the variance
    bootstrap_betas_var = np.maximum(bootstrap_betas_sum_of_sq - (bootstrap_betas_mean ** 2), 0.0)
    
    # Create a dictionary with results
    results = dict(returns=np.array(bootstrap_rtns)[:,idx_revert], 
                   betas_mean=np.array(bootstrap_betas_mean)[idx_revert,:],
                   betas_var=np.array(bootstrap_betas_var)[idx_revert,:])
    results['betas_zscore'] = results['betas_mean'] / np.sqrt(results['betas_var'] + 1e-10)
    return results

def _get_asset_excess_returns(asset_rtns, rfr_rtns):
    if isinstance(rfr_rtns, np.ndarray):
        return asset_rtns - rfr_rtns
    else:
        ar, rf = asset_rtns.align(rfr_rtns, axis=0, join='inner')
        return ar - rf.values
    
def _generate_bootstrap_locations_for_regression_resampling(rand_state, n_rows, n_cols, first_rows):
    bootstrap_locs = -1 * np.ones((n_rows, n_cols), dtype=int)
    left = right = 0
    for right in range(n_cols):
        if right == n_cols - 1 or first_rows[right] != first_rows[right+1]:
            start = first_rows[left]
            end = first_rows[right+1] if right < n_cols - 1 else n_rows
            rand_locs = rand_state.choice(range(start, end), end - start)[:,np.newaxis]
            bootstrap_locs[start:end, :right+1] = rand_locs
            left = right + 1
    return bootstrap_locs

def _order_columns_by_first_nan(vals):    
    # Find the first non-NaN rows
    first_rows = _find_first_non_nan_rows(vals)
    
    # Order the columns so the rows that start earliest are on the left
    sort_idx = np.argsort(first_rows)
    ordered_vals = vals[:,sort_idx]
    idx_revert = np.argsort(sort_idx)
    return ordered_vals, idx_revert

def _find_first_non_nan_rows(mtx):    
    """ Find the first non-NaN rows in a numpy matrix. """
    first_rows = []
    for j in range(mtx.shape[1]):
        if np.all(np.isnan(mtx[:,j])):
            first_rows.append(mtx.shape[0])
        else:
            first_rows.append(np.where(~np.isnan(mtx[:,j]))[0][0])
    return np.array(first_rows, dtype=int)
    
def _get_date_aligned_numpy(rfr_rtns, factor_rtns, asset_rtns):
    if isinstance(rfr_rtns, np.ndarray):
        return rfr_rtns, factor_rtns, asset_rtns
    else:
        common_dates = pd.DatetimeIndex.intersection(rfr_rtns.index, asset_rtns.index)
        common_dates = pd.DatetimeIndex.intersection(common_dates, factor_rtns.index)
        rfr_rtns = rfr_rtns.loc[common_dates].values
        factor_rtns = factor_rtns.loc[common_dates].values
        asset_rtns = asset_rtns.loc[common_dates].values
        return rfr_rtns, factor_rtns, asset_rtns

def _check_time_series_consistency(ts, *args):
    T = ts.shape[0]
    for ats in args:
        if isinstance(ts, np.ndarray) and not isinstance(ats, np.ndarray):
            raise ValueError('If one time series is a numpy array, then they all must be numpy arrays.')
        elif isinstance(ts, (pd.Series, pd.DataFrame)) and not isinstance(ats, (pd.Series, pd.DataFrame)):
            raise ValueError('If one time series is a pandas object, then they all must be pandas objects.')
    return True

def _get_n_periods_per_year(freq, rfr_rtns_ts):
    # Get the number of periods per year so we can annualize returns/covariance
    if isinstance(rfr_rtns_ts, (pd.Series, pd.DataFrame)):
        if freq is not None:
            raise ValueError('The frequency argument should only be used if the inputs are numpy arrays.')
        else:
            n_periods_per_year = rfr_rtns_ts.fts.n_periods_per_year
    elif freq is not None:
        n_periods_per_year = pyfintools.tools.fts.get_periods_per_year(freq)
    else:
        raise ValueError('The frequency argument must be used if the inputs are pandas arrays.')
    return n_periods_per_year, freq

def _extract_size_info(size):
    if isinstance(size, tuple):
        if len(size) == 2:
            n_steps, n_paths = size
        elif len(size) == 1:
            n_steps = size[0]
            n_paths = 1
        else:
            raise ValueError('Unsupported dimensions for argument "size".')
    elif isinstance(size, int):
        n_steps = size
        n_paths = 1
    else:
        raise ValueError('Unsupported input type "{}" for argument "size."'.format(size.__class__))
    return n_steps, n_paths

def _align_time_series(rfr, factor, asset, ts_type):
    r, f, a = _align_time_series_frequency(rfr, factor, asset, ts_type)
    return _align_time_series_dates(r, f, a)

def _align_time_series_frequency(rfr, factor, asset, ts_type):
    if isinstance(rfr, np.ndarray):
        if rfr.shape[0] != factor.shape[0] or rfr.shape[0] != asset.shape[0]:
            raise ValueError('All time series dimensions must have the same length if the inputs are numpy arrays.')
    elif ts_type == 'returns':
        frequencies = list(set([pd.infer_freq(ts.index) for ts in [rfr, factor, asset]]))
        if len(frequencies) == 1:
            freq = frequencies[0].lower()
        elif len(frequencies) == 2 and all([f in ['BM', 'SM', 'M'] for f in frequencies]):
            freq = 'm'
        elif len(frequencies) == 2 and all([f in ['D', 'B'] for f in frequencies]):
            freq = 'd'
        else:
            raise NotImplementedError('Support for incompatible frequencies {} is not yet implemented.'.format(frequencies))
        rfr = rfr.resample(freq).last()
        factor = factor.resample(freq).last()
        asset = asset.resample(freq).last()
    else:
        raise NotImplementedError(f'Handling time series of type {ts_type} is not yet implemented.')

    return rfr, factor, asset

def _align_time_series_dates(rfr, factor, asset):
    r, _ = rfr.align(factor, axis=0, join='outer')
    a, _ = asset.align(factor, axis=0, join='outer')
    return r.loc[factor.index], factor, a.loc[factor.index]
    