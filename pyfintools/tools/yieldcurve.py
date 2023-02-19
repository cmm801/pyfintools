""" Define classes for working with yield data, and converting between spot and forward yields.

    Main classes are ZeroCurve and ForwardCurve. Both allow conversion to the other type of yield,
    and both provide access to a method 'get_implied_path', which calculates the future yield
    surface based on the expectations hypothesis (Recall that the expectations hypothesis (EH) says
    that market participants' best estimate for the future evolution of the yield curve is 
    contained in the forward rates.
"""

import numpy as np
import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt

import pyfintools.constants
import pyfintools.tools.freq


class IRCurve(object):
    rate_type = ''
    
    def __init__(self, values, tenor_in_years):
        super(IRCurve, self).__init__()
        
        # Initialize some variables that will be set later
        self._tenor_in_years = None
        self.tenor_spacing_in_years = None
        self.values = None

        self.values = values
        self.tenor_in_years = tenor_in_years
        # Set the values from the input arguments
        assert self.values.shape == tenor_in_years.shape, \
                        'Tenors and rates must be of the same size.'

    @property
    def values(self):
        return self._values
    
    @values.setter
    def values(self, values):
        if isinstance(values, np.ndarray):
            self._values = values.flatten().astype(float)
        else:
            self._values = np.array(values, dtype=float)
        
    @property
    def tenor_in_years(self):
        return self._tenor_in_years
    
    @tenor_in_years.setter
    def tenor_in_years(self, tenors):
        diffs = tenors[1:] - tenors[:-1]
        if not diffs.size:
            self.tenor_spacing_in_years = None
            self._tenor_in_years = np.array(tenors, dtype=float)
        elif not np.isclose(np.min(diffs), np.max(diffs)):
            self._set_tenor_in_years_with_non_unique_spacings(tenors)
        elif not self._is_integer(12 * np.mean(diffs)):
            raise ValueError('The spacing between tenors must all be multiples of 1/12-th')
        else:
            self.tenor_spacing_in_years = np.mean(diffs)
            if isinstance(self._tenor_in_years, np.ndarray):
                self._tenor_in_years = np.array(tenors.flatten(), dtype=float)
            else:
                self._tenor_in_years = np.array(tenors, dtype=float)

    @property
    def tenor_in_months(self):
        return (self._tenor_in_years * 12).astype(int)

    @property
    def size(self):
        return self.values.size

    def get_implied_path(self, data_frequency='Y', adjust_for_convexity=False, yield_exp_vols=None):
        raise NotImplementedError('Must be implemented in the subclass.')

    def get_implied_path_ts(self, pricing_date, data_frequency='Y', adjust_for_convexity=False, yield_exp_vols=None):
        # Get the matrix of the implied path for rates
        rate_list = self.get_implied_path(data_frequency=data_frequency,
                                          adjust_for_convexity=adjust_for_convexity, 
                                          yield_exp_vols=yield_exp_vols)
        rate_mtx = np.vstack([x.values.reshape(1,-1) for x in rate_list])
        
        # Get the future dates correspoding to the future rates
        future_dates = pd.date_range(pricing_date, periods=rate_mtx.shape[0], freq=data_frequency.upper())
        
        # Combine the future dates and rates and return a DataFrame
        return pd.DataFrame(rate_mtx, index=future_dates, columns=self.tenor_in_years)
        
    def plot(self, ax=None, **kwargs):
        if ax is None:
            plt.plot(self.tenor_in_years, self.values, **kwargs)
        else:
            ax.plot(self.tenor_in_years, self.values, **kwargs)

    def _set_tenor_in_years_with_non_unique_spacings(self, tenors):
        self.tenor_spacing_in_years = np.nan
        self._tenor_in_years = np.array(tenors, dtype=float)

    def _is_integer(self, val):
        return np.isclose(0, np.abs(np.round(val, 0) - val), atol=1e-6)    

    def _interpolate(self, target_tenors):
        if isinstance(target_tenors, pyfintools.constants.NUMERIC_DATA_TYPES):
            target_tenors = np.array([target_tenors], dtype=float)
        else:
            target_tenors = np.array(target_tenors, dtype=float)
            
        interp_panel = scipy.interpolate.splrep(self.tenor_in_years, self.values, s=0)
        interp_vals = scipy.interpolate.splev(target_tenors, interp_panel, der=0)
        return self.__class__(interp_vals, target_tenors)

class ZeroCurve(IRCurve):
    rate_type = 'zero'
    
    def get_zeros(self, target_tenors=None):
        if target_tenors is None:
            return ZeroCurve(self.values, self.tenor_in_years)
        else:
            return self._interpolate(target_tenors)

    def get_forwards(self, target_tenors=None):
        tenor_diffs = self.tenor_in_years[1:] - self.tenor_in_years[:-1]
        if not np.isclose(np.min(tenor_diffs), np.max(tenor_diffs)):
            raise ValueError('The tenors provided in the ForwardCurve must be equally spaced from one another.')
        tenor_spacing_in_years = np.mean(tenor_diffs)
        
        # We keep the first yield unchanged, but then take the difference of all of the others to get the forwards
        forward_values = np.nan * np.ones((self.tenor_in_years.size,), dtype=float)
        forward_values[0] = self.tenor_in_years[0] * self.values[0]
        forward_values[1:] = (self.tenor_in_years[1:] + tenor_spacing_in_years) * self.values[1:] \
                           -  self.tenor_in_years[1:] * self.values[:-1]
        fwd_curve = ForwardCurve(forward_values / tenor_spacing_in_years, self.tenor_in_years)
        
        # If we are asked for a specific set of tenors, then we need to interpolate
        if target_tenors is None:
            return fwd_curve
        else:
            return fwd_curve.get_forwards(target_tenors)

    def get_implied_path(self, data_frequency='Y', adjust_for_convexity=False, yield_exp_vols=None):
        fc = self.get_forwards()
        implied_fwd_path = fc.get_implied_path(data_frequency=data_frequency,
                                               adjust_for_convexity=adjust_for_convexity,
                                               yield_exp_vols=yield_exp_vols)
        zero_values = []
        for fwd in implied_fwd_path:
            zero_values.append(fwd.get_zeros())
        return zero_values


class ForwardCurve(IRCurve):
    rate_type = 'forward'
    
    def __init__(self, values, tenor_in_years):
        super(ForwardCurve, self).__init__(values, tenor_in_years)
    
    @property
    def tenor_in_months(self):
        return (self._tenor_in_years * 12).astype(int)
        
    @classmethod
    def from_zero_yields(cls, yield_values, tenor_in_years):
        yc_obj = ZeroCurve(yield_values, tenor_in_years)
        return yc_obj.get_forwards()
    
    def get_forwards(self, target_tenors=None):
        if target_tenors is None:
            return ForwardCurve(self.values, self.tenor_in_years)
        else:
            return self._interpolate(target_tenors)
    
    def get_zeros(self, target_tenors=None):
        if target_tenors is not None:
            fwd_curve = self.get_forwards(target_tenors)
        else:
            fwd_curve = self
            
        zero_values = np.cumsum(fwd_curve.values) * fwd_curve.tenor_spacing_in_years / fwd_curve.tenor_in_years
        return ZeroCurve(zero_values, fwd_curve.tenor_in_years)

    def get_implied_path(self, data_frequency='Y', adjust_for_convexity=False, yield_exp_vols=None):
        # Get the range of tenors based on the data frequency
        T = 1/pyfintools.tools.freq.get_periods_per_year(data_frequency)
        max_tenor = self.tenor_in_years.max()
        target_tenors = np.arange(T, max_tenor + T/2, T)
        
        future_forwards = [self.get_forwards(target_tenors)]
        for t in range(self.size-1):
            prev_obj = future_forwards[-1]
            new_fwd_vals = np.hstack([prev_obj.values[1:], np.nan])
            new_obj = ForwardCurve(new_fwd_vals, target_tenors)

            # Adjust for convexity if requested
            if adjust_for_convexity:
                prev_zero_values = prev_obj.get_zeros().values
                new_obj._adjust_forwards_for_convexity(prev_zero_values, yield_exp_vols)

            future_forwards.append(new_obj)

        # Combine the rates into a 2d array before returning
        return future_forwards

    def _adjust_forwards_for_convexity(self, prev_zero_values, yield_exp_vols):
        # Adjust the forwards for convexity, if desired
        convexity = (self.tenor_in_years ** 2 + self.tenor_in_years) * np.exp(-2 * prev_zero_values)
        cvx_adj = 0.5 * (convexity * (yield_exp_vols ** 2)).flatten()
        diff_in_cvx_adj = np.hstack([cvx_adj[1:] - cvx_adj[:-1], np.nan])
        self.values += diff_in_cvx_adj

    def _set_tenor_in_years_with_non_unique_spacings(self, tenors):
        raise ValueError('All spacings between tenors must be unique to use the ForwardCurve object.')
