""" A module containing a number of classes and functions for performing statistical techniques.

    This module contains a class for performing linear regressions, and also code for
        unsmoothing auto-correlated time series (using the Geltner technique)
    
"""

import numpy as np
import pandas as pd
import statsmodels.regression.linear_model
from abc import ABC, abstractmethod
import sklearn.linear_model


class AbstractRegression(ABC):
    def __init__(self, fit_intercept=True):
        super(AbstractRegression, self).__init__()
        self.fit_intercept = fit_intercept
 
    @property
    def fit_intercept(self):
        return self._fit_intercept

    @fit_intercept.setter
    def fit_intercept(self, fi):
        self._fit_intercept = fi
        self.invalidate_cache()
     
    @property
    def results(self):
        return self._results
    
    @property
    def intercept(self):
        if self.results is None:
            return 0
        else:
            return self.results.intercept
    
    @property
    def coefs(self):
        if self.results is None:
            return None
        else:
            return self.results.coefs

    @property
    def yhat(self):
        if self.results is None:
            return None
        else:
            return self.results.yhat
    
    @property
    def residual(self):
        if self.results is None:
            return None
        else:
            return self.results.residual
    
    @abstractmethod
    def _fit_core(self, X, y, sample_weight=None):
        raise NotImplementedError('Must be implemented by subclass.')

    def fit(self, X, y, sample_weight=None):
        X_eff, y_eff = self._align_inputs(X, y)
        if self.fit_intercept:
            X_eff = np.hstack([np.ones((X_eff.shape[0], 1), dtype=float), X_eff])

        self._results = self._fit_core(X_eff, y_eff, sample_weight=sample_weight)
        self._results.X = X
        self._results.y = y
        return self._results

    def predict(self, X):
        if self.results is None:
            raise ValueError('The "fit" method must be called before "predict" can be called.')

        return self.intercept + (X @ np.array(self.coefs, dtype=float))

    def invalidate_cache(self):
        self._results = None

    def _align_inputs(self, X, y):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X)
        elif X.ndim == 1:
            X = X.reshape(-1, 1)
        return X, y


class RegressionResults(statsmodels.regression.linear_model.RegressionResultsWrapper):
    def __init__(self, _results, fit_intercept):
        self._results = _results
        self.fit_intercept = fit_intercept
    
    @classmethod
    def from_wrapper(cls, wrapper, fit_intercept):
        return RegressionResults(wrapper._results, fit_intercept=fit_intercept)
    
    @property
    def coefs(self):
        if self.fit_intercept:
            return self.params[1:]
        else:
            return self.params

    @property
    def intercept(self):
        if self.fit_intercept:
            return self.params[0]
        else:
            return 0.0

    @property
    def yhat(self):
        return self.intercept + (self.X @ np.array(self.coefs, dtype=float))
    
    @property
    def residual(self):
        return self.y - self.yhat


class OLS(AbstractRegression):
    def _fit_core(self, X, y, sample_weight=None):
        model = statsmodels.regression.linear_model.OLS(endog=y, exog=X, sample_weight=sample_weight)
        raw_res = model.fit()
        return RegressionResults.from_wrapper(raw_res, fit_intercept=self.fit_intercept)


def geltner_unsmooth(input_series, skipna=True):
    """ Use Geltner (1993) Unsmoothing method to unsmooth the returns.
        The original Geltner paper is behind a pay-wall, but the formula can be found in Appendix 2 of:
        https://am.jpmorgan.com/blobcontent/1383399455015/83456/II-Hedge-fund-volatility_FINAL.pdf
    """
    if input_series.ndim == 1:
        input_series = input_series.reshape(-1, 1)
    elif input_series.ndim > 2:
        raise ValueError('Only supported for 2-d numpy arrays.')

    lag_coefs = []
    Z_adj_list = []    
    for j in range(input_series.shape[1]):
        Z = input_series[:,j]

        # Calculate the auto-correlated component
        y = Z[1:]
        x = Z[:-1]
        
        # Remove rows with missing data, if requested to do so
        if skipna:
            idx = ~np.isnan(x) & ~np.isnan(y)
            x = x[idx]
            y = y[idx]

        # Create the regression model
        model = OLS(fit_intercept=True)
        res = model.fit(x, y)
        beta = float(res.coefs)

        # Subtract the auto-correlated component
        Z_adj_j = np.hstack([Z[0], (Z[1:] - beta * Z[:-1]) / (1 - beta)]).reshape(-1, 1)
        
        # Store the results
        lag_coefs.append(beta)
        Z_adj_list.append(Z_adj_j)

    # Calculate the unsmoothed time series
    Z_uns = np.hstack(Z_adj_list)
    return Z_uns

def geltner_unsmooth_rolling(input_series, window, skipna=True, max_beta=0.75, smoothing=0.9):
    """ Generalize the Geltner method to unsmooth over rolling windows.
    
        Inspired by Geltner (1993) Unsmoothing method to unsmooth the returns.
        The original Geltner paper is behind a pay-wall, but the formula can be found in Appendix 2 of:
        https://am.jpmorgan.com/blobcontent/1383399455015/83456/II-Hedge-fund-volatility_FINAL.pdf
        
        Some additional arguments have been added to keep the unsmoothed results from blowing up - 
            this was especially important for the NCREIF Private Real Estate Index, where the 
            one-lag autocorrelation was close to 0.85.
            A 'smoothing' parameter was added to exponentiall smooth the one-lag beta values from 
            one window to the next.
            
        Note: for the initial observations, with index < window size, rather than calculate the
            autocorrelation coefficient with a few data points, we just use the full-sample
            autocorrelation. Then, the autocorrelation coefficient evolves according to the smoothing
            parameter.
        
        Arguments:
            input_series: (nump array) A numpy array, where each column will separately be unsmoothed
            window: (int) the number of observations to include in each window when unsmoothing
            skipna: (bool) whether to ignore any NaN values in the calculation. Default is True.
            max_beta: (float) this is a cap on the allowed one-lag beta. The reason for this is that
                the unsmoothed returns are obtained by dividing the original returns by (1-beta). 
                Therefore as beta goes to 1, the unsmoothed result becomes unstable.
            smoothing: (float) this is the exponential smoothing parameter of the one-lag beta from 
                one window to the next. If smoothing == 0, then the autocorrelation coefficient
                from a given window is used without adjustment. If smoothing == 1, then the beta 
                is never updated and this approach becomes equivalent to the original (non-rolling) 
                Geltner unsmoothing method.
        
    """
    if input_series.ndim == 1:
        input_series = input_series.reshape(-1, 1)
    elif input_series.ndim > 2:
        raise ValueError('Only supported for 2-d numpy arrays.')

    Z_adj_list = []
    T = input_series.shape[0]
    for j in range(input_series.shape[1]):
        Z = input_series[:,j]

        model = sklearn.linear_model.LinearRegression()
        x = Z[:-1].ravel()
        y = Z[1:]
        if skipna:
            idx = ~np.isnan(x) & ~np.isnan(y)
            x = x[idx]
            y = y[idx]
        
        res = model.fit(X=x.reshape(-1, 1), y=y)
        beta0 = min(max_beta, res.coef_[0])
        beta = beta0

        adj_rtns = [np.nan]
        for t in range(1, T):
            if t < window:
                r_uns = (Z[t] - beta * Z[t-1]) / (1 - beta)
                adj_rtns.append(r_uns)
            else:
                y = Z[t-window+1:t+1]
                x = Z[t-window:t]

                # Remove rows with missing data, if requested to do so
                if skipna:
                    idx = ~np.isnan(x) & ~np.isnan(y)
                    x = x[idx]
                    y = y[idx]

                # Create the regression model
                model = sklearn.linear_model.LinearRegression()
                res = model.fit(X=x.reshape(-1, 1), y=y)

                # Update the beta on the rolling interval
                beta_update = res.coef_[0]
                beta = min(max_beta, smoothing * beta + (1-smoothing) * beta_update)

                # Unsmooth the current period return
                r_uns = (y[-1] - beta * x[-1]) / (1 - beta)
                adj_rtns.append(r_uns)
                
        # Store the unsmoothed returns for the j-th column
        Z_adj_list.append(np.array(adj_rtns, dtype=float))

    # Combine the unsmoothed time series into a single object
    Z_uns = np.vstack(Z_adj_list).T
    return Z_uns
