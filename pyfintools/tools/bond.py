""" Contains classes for calculating the prices of a single bond or portfolio of bonds.
"""

import collections
import bisect
import scipy.optimize
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

import secdb.constants
import secdb.tools.freq


class ZeroCurveAbstract(object):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def _get_zero_rates(self, pricing_date, tenor):
        raise NotImplementedError('Needs to be implemented by subclass')

    def calc_discounted_value(self, pricing_date, cashflows, n_years):
        if not isinstance(cashflows, secdb.constants.NUMERIC_DATA_TYPES):
            cashflows = np.array(cashflows, dtype=float)

        if not isinstance(n_years, secdb.constants.NUMERIC_DATA_TYPES):
            n_years = np.array(n_years, dtype=float)

        zero_rates = self._get_zero_rates(pricing_date, n_years)
        discounted_vals = cashflows / np.power(1 + zero_rates, n_years)
        return discounted_vals        
        
        
class ZeroCurveConstant(ZeroCurveAbstract):
    def __init__(self, yield_val):
        super().__init__()
        self.yield_val = yield_val
        
    def _get_zero_rates(self, pricing_date, tenor):
        return self.yield_val


class AbstractBond(object):
    def __init__(self, *args, **kwargs):
        super(AbstractBond, self).__init__()
        
    @abstractmethod
    def cashflows(self):
        pass

    @abstractmethod
    def get_cashflows(self, start=None, end=None, closed='both'):
        pass

    @abstractmethod
    def calc_accrued_interest(self, pricing_date, zero_yield_obj):
        pass

    @abstractmethod        
    def invalidate_cache(self):
        pass

    def calc_clean_price(self, pricing_date, zero_yield_obj):
        dirty_price = self.calc_price(pricing_date, zero_yield_obj)
        accrued_interest = self.calc_accrued_interest(pricing_date, zero_yield_obj)
        clean_price = dirty_price - accrued_interest
        return clean_price, accrued_interest

    def calc_price(self, pricing_date, zero_yield_obj):
        cf = self.get_cashflows(pricing_date)
        return calc_price_from_cashflows(cf, cf.index, pricing_date, zero_yield_obj)

    def calc_convexity(self, pricing_date):
        cf = self.get_cashflows(pricing_date)        
        years_to_maturity = secdb.tools.freq.get_years_between_dates(pricing_date, cf.index)
        return (np.power(years_to_maturity, 2) * cf.values).sum() / cf.values.sum()
        
    def calc_duration_mod(self, pricing_date, zero_yield_obj, use_ytm=False, 
                          compounding_freq=secdb.constants.CONTINUOUS_COMPOUNDING, verbose=False):
        """ Calculate the modified duration.
            The modified duration is the linear sensitivity of the bond price to small changes
            in the yield-to-maturity. The formula is D_mod = D_macauley / (1 + Y_YTM)
        """
        duration_mac = self.calc_duration_mac(pricing_date, zero_yield_obj, use_ytm=use_ytm)
        ytm = self.calc_YTM(pricing_date, zero_yield_obj, compounding_freq=compounding_freq, verbose=verbose)
        return duration_mac / (1 + ytm)
        
    def calc_duration_mac(self, pricing_date, zero_yield_obj, use_ytm=False):
        """ Calculate the Macauley duration. 
            The Macauley duration is the time-weighted average of discounted cash flows.
            We use the zero curve to discount the cash flows.
        """
        cf = self.get_cashflows(pricing_date)
        years_to_maturity = secdb.tools.freq.get_years_between_dates(pricing_date, cf.index)
        
        if cf.size == 1:
            # For zero coupon bonds, the duration is just the time to the final payout
            return secdb.tools.freq.get_years_between_dates(pricing_date, cf.index[0])
        else:
            if use_ytm:
                YTM = self.calc_YTM(pricing_date, zero_yield_obj)
                D = np.exp(-YTM * years_to_maturity)
                discounted_vals = cf * D
            else:
                discounted_vals = zero_yield_obj.calc_discounted_value(pricing_date, cf, years_to_maturity)

            time_wtd_vals = years_to_maturity * discounted_vals
            return np.sum(time_wtd_vals) / np.sum(discounted_vals)            

    def calc_YTM(self, pricing_date, zero_yield_obj, verbose=False, compounding_freq=secdb.constants.CONTINUOUS_COMPOUNDING):
        cashflows = self.get_cashflows(pricing_date)
        prc = self.calc_price(pricing_date, zero_yield_obj)
        n_years_to_mty = secdb.tools.freq.get_years_between_dates(pricing_date, cashflows.index)

        def min_fun(YTM):
            if compounding_freq == secdb.constants.CONTINUOUS_COMPOUNDING:
                D = np.exp(-YTM * n_years_to_mty)
            elif isinstance(compounding_freq, int):
                D = 1 / np.power(1 + YTM/compounding_freq, compounding_freq * n_years_to_mty)
            else:
                raise ValueError(f'Unsupported compounding frequency: {compounding_freq}')
            return (prc - np.sum(cashflows.values * D)) ** 2

        # Use the current yield as the initial guess when solving the optimization problem
        T_F = max(n_years_to_mty)
        ytm0 = float(zero_yield_obj.loc[pricing_date].get_yields(T_F).values)
        res = scipy.optimize.minimize(min_fun, ytm0, bounds=[(0, 1)], tol=1e-4)
        if verbose:
            print(res)

        if res.success:
            return res.x[0]
        else:
            raise ValueError('Unsuccessful calculation of YTM')
    
    
class Bond(AbstractBond):
    _counters = []

    def __init__(self, settlement_date, maturity_date, coupon, coupon_freq=2, notional=100):
        super(Bond, self).__init__()
            
        # Initialize the cache for storing the time series of cash flows
        self.invalidate_cache()
        
        self._coupon_freq = None
        self._notional = None

        self.settlement_date = pd.Timestamp(settlement_date).floor('D')        
        self.maturity_date = pd.Timestamp(maturity_date).floor('D')
        self.coupon = coupon
        self.coupon_freq = coupon_freq
        self.notional = notional

        # Set the unique ID for the Bond
        self.uid = 0 if not self._counters else self._counters[-1] + 1
        self._counters.append(self.uid)

    # Implement abstractmethod
    def invalidate_cache(self):
        self._unscaled_coupons = None
    
    # Implement abstractmethod
    @property
    def cashflows(self):
        cf = self.notional * self.unscaled_cashflows
        if np.any(np.isnan(cf.values)):
            raise ValueError('Cashflows cannot be NaN')
        else:
            return cf
    
    @property
    def unscaled_coupons(self):
        if self._unscaled_coupons is None:
            if np.isclose(0, self.coupon_freq):
                # No coupons paid out for a zero-coupon bond
                self._unscaled_coupons = pd.Series([], index=[])
            else:
                cf_dates = [self.maturity_date]
                n_months = int(12 / self.coupon_freq)
                prev_date = cf_dates[-1] - pd.DateOffset(months=n_months)
                while prev_date > self.settlement_date:
                    cf_dates.append(prev_date)
                    prev_date = cf_dates[-1] - pd.DateOffset(months=n_months)

                cf_dates = cf_dates[::-1]
                cf_amounts = 1/self.coupon_freq * np.ones((len(cf_dates),), dtype=float)

                self._unscaled_coupons = pd.Series(cf_amounts, index=pd.DatetimeIndex(cf_dates), name=self.uid)
                self._unscaled_coupons.sort_index(inplace=True)
        return self._unscaled_coupons

    @property
    def unscaled_cashflows(self):
        unscaled_cf = self.coupon * self.unscaled_coupons.copy()
        if self.maturity_date in unscaled_cf.index:
            unscaled_cf.loc[self.maturity_date] += 1
        elif np.isclose(0, self.coupon_freq):
            unscaled_cf = pd.Series([1.0], index=pd.DatetimeIndex([self.maturity_date]), name=self.uid)
        else:
            raise ValueError('The unscaled cashflows should not be empty except for zero coupon bonds.')
        return unscaled_cf
    
    @property
    def scaled_coupons(self):
        return self.notional * self.coupon * self.unscaled_coupons
    
    @property
    def coupon_freq(self):
        return self._coupon_freq
    
    @coupon_freq.setter
    def coupon_freq(self, freq):
        if not isinstance(freq, int):
            raise ValueError('Coupon frequency must be an integer, representing # of payments per year.')
        if freq and not 12 % freq == 0:
            raise ValueError('Coupon frequency must be 0 or else a divisor of 12.')
        else:
            if self.coupon_freq is not None and not np.isclose(self.coupon_freq, freq):
                self.invalidate_cache()
            self._coupon_freq = freq

    @property
    def notional(self):
        return self._notional
    
    @notional.setter
    def notional(self, val):
        if not isinstance(val, secdb.constants.NUMERIC_DATA_TYPES):
            raise ValueError('Notional must be a float or integer value.')
        else:
            self._notional = val
            
    # Implement abstractmethod            
    def calc_accrued_interest(self, pricing_date, zero_yield_obj):
        next_payment_amt, next_payment_date = self._get_next_coupon(pricing_date, closed=True)
        if next_payment_amt is None:
            # If there are no further payments...
            return 0.0
        elif next_payment_date == pricing_date:
            # The next payment is fully accrued on the pricing date
            return next_payment_amt
        else:
            _, prev_payment_date = self._get_previous_coupon(pricing_date, closed=False)
            if prev_payment_date is None:
                if pricing_date < self.settlement_date:
                    raise ValueError('Cannot use a pricing date before the settlement date of the bond.')
                else:
                    raise ValueError('Error in obtaining the previous coupon payment date.')

            # Find the number of years between the previous and next coupon payments
            years_from_prev_payment = secdb.tools.freq.get_years_between_dates(prev_payment_date, pricing_date)
            years_to_next_payment = secdb.tools.freq.get_years_between_dates(pricing_date, next_payment_date)
            
            # Calculate the discounted value of the future coupon payment
            disc_val_of_next_cashflow = float(zero_yield_obj.calc_discounted_value(pricing_date, next_payment_amt,
                                                                                   years_to_next_payment))

            # Calculate the proportion of the next coupon payment that has been accrued
            years_btwn_payments = years_from_prev_payment + years_to_next_payment            
            proportion_accrued = 1 - years_to_next_payment / years_btwn_payments
            return proportion_accrued * disc_val_of_next_cashflow

    def calc_par_yield(self, pricing_date, zero_yield_obj):
        if np.isclose(0, self.coupon_freq):
            raise ValueError('Cannot calculate the par yield for a zero coupon bond.')
        else:
            cash_flows = self.get_cashflows(pricing_date).copy()
            coupon_flows = cash_flows
            idx_last = coupon_flows.index[-1]
            coupon_flows[-1] -= self.notional

            dcf_principal = calc_price_from_cashflows(self.notional, self.maturity_date, pricing_date, zero_yield_obj) 
            dcf_coupons = calc_price_from_cashflows(coupon_flows.values, coupon_flows.index, 
                                                    pricing_date, zero_yield_obj) 

            # Calculate the par yield from the discounted components and the notional
            dcf_coupons_unscaled = dcf_coupons / self.coupon
            par_yield = (self.notional - dcf_principal) / dcf_coupons_unscaled
            assert dcf_coupons_unscaled > 0, 'Invalid'
            return par_yield

    # Implement abstractmethod
    def get_cashflows(self, start=None, end=None, closed='both'):
        if start is None:
            start = self.settlement_date
            
        if end is None:
            end = self.maturity_date
                
        # Get the cash flows on or after the pricing date from the Cache
        period_flows = self.cashflows.loc[start:end]
        if closed == 'right':
            cf = period_flows.loc[period_flows.index != start]
        elif closed == 'left':
            cf = period_flows.loc[period_flows.index != end]
        elif closed == 'neither':
            cf = period_flows.loc[(period_flows.index != start) & (period_flows.index != end)]
        elif closed == 'both':
            cf = period_flows
        else:
            raise ValueError(f'Unsupported value for input argument "closed" - {closed}')

        return cf
            
    def _get_next_coupon(self, pricing_date, closed):
        cf = self.scaled_coupons
        coupon_amt = coupon_date = None
        if cf.size:
            if closed:
                idx = cf.index[pricing_date <= cf.index]
            else:
                idx = cf.index[pricing_date < cf.index]
            
            if idx.size:
                coupon_amt = cf[idx[0]]
                coupon_date = idx[0]
                
        return coupon_amt, coupon_date
            
    def _get_previous_coupon(self, pricing_date, closed):
        cf = self.scaled_coupons
        coupon_amt = coupon_date = None
        if cf.size:
            if closed:
                idx = cf.index[pricing_date >= cf.index]
            else:
                idx = cf.index[pricing_date > cf.index]
            
            if idx.size:
                coupon_amt = cf[idx[-1]]
                coupon_date = idx[-1]

        if coupon_amt is None and pricing_date >= self.settlement_date:
            coupon_amt = 0
            _, next_coupon_date = self._get_next_coupon(pricing_date, closed=closed)
            n_months = int(12 / self.coupon_freq)
            coupon_date = next_coupon_date - pd.DateOffset(months=2 * n_months)

        return coupon_amt, coupon_date


class BondPortfolio(AbstractBond):
    def __init__(self, bonds=[]):
        super(BondPortfolio, self).__init__()
        self.bonds = bonds

    # Implement abstractmethod        
    @property
    def cashflows(self):
        cf_list = []
        for bd in self.bonds:
            one_bond_cf = bd.cashflows
            cf_list.append(one_bond_cf)
        return pd.concat(cf_list, axis=0)

    # Implement abstractmethod
    def get_cashflows(self, start=None, end=None, closed='both'):
        cf_list = []
        for bd in self.bonds:
            one_bond_cf = bd.get_cashflows(start=start, end=end, closed=closed)
            cf_list.append(one_bond_cf)
        return pd.concat(cf_list, axis=0)

    # Implement abstractmethod
    def calc_accrued_interest(self, pricing_date, zero_yield_obj):
        interest = 0.0
        for bd in self.bonds:
            interest += bd.calc_accrued_interest(pricing_date=pricing_date, zero_yield_obj=zero_yield_obj)
        return interest

    # Implement abstractmethod
    def invalidate_cache(self):
        for bd in self.bonds:
            bd.invalidate_cache()



def calc_price_from_cashflows(cf_amounts, cf_dates, pricing_date, zero_yield_obj):
    """ Discount a set of cash flows.
        Arguments:
          cf_amounts: an array of float values indicating future cashflows.
          cf_dates: the dates on which the corresponding cf_amounts will be sitributed.
          pricing_date: the date on which the price is to be calculated
          zero_yield_obj: either a YieldCurve object representing zero yields, or a float representing the YTM
          """
    years_to_maturity = secdb.tools.freq.get_years_between_dates(pricing_date, cf_dates)
    discounted_vals = zero_yield_obj.calc_discounted_value(pricing_date, cf_amounts, years_to_maturity)
    return np.sum(discounted_vals)
