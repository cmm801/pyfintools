""" Contains classes for estimating equity prices (e.g. Implied ERP).
"""


import numpy as np
import copy
import scipy.optimize

import pyfintools.tools.utils


class ImpliedERP(object):
    """ Calculate the Implied Equity Risk Premium (ERP) using Damadaran's methodology.
        The model uses a 2-period Dividend Discount model, where the first period uses
        analyst forecasts of earnings growth and the second period assumes that earnings
        grow at the long-term discount rate.

        Arguments:
            initial_earnings: (float) Initial Earnings (e.g. 12-month trailing earnings)
            initial_distributions: (float) Initial Cash distributions (e.g. 12m-month trailing 
                                                distributions of dividends + net buybacks)
            adjust_payout_sust: (bool) whether or not to adjust the payout ratio to its sustainable level 
                            of [1 - growth rate / ROE] by the end of Period 1
            years_in_period_1: (float) number of years in Period 1
    """
    def __init__(self, market_value, initial_earnings, initial_distributions, 
                 exp_period_1_earnings_growth, exp_period_2_earnings_growth, 
                 long_term_ERP, long_term_ROE, current_rfr, 
                 adjust_payout_sust=True, years_in_period_1=5,
                 adjust_for_crash=False, prop_earnings_recouped=0.8, earnings_drop=0.25):
        super(ImpliedERP, self).__init__()

        self.market_value = market_value
        self.initial_earnings = initial_earnings
        self.initial_distributions = initial_distributions
        self.exp_period_1_earnings_growth = exp_period_1_earnings_growth
        self.exp_period_2_earnings_growth = exp_period_2_earnings_growth
        self.long_term_ERP = long_term_ERP
        self.long_term_ROE = long_term_ROE
        self.current_rfr = current_rfr
        self.years_in_period_1 = years_in_period_1        
        self.adjust_payout_sust = adjust_payout_sust

        # Additional variables to help adjust for a crash scenario
        self.adjust_for_crash = adjust_for_crash
        self.prop_earnings_recouped = prop_earnings_recouped
        self.earnings_drop = earnings_drop
        
    def copy(self):
        return copy.deepcopy(self)

    @property
    def initial_payout_ratio(self):
        return self.initial_distributions / self.initial_earnings
        
    @property
    def sustainable_payout_ratio(self):
        return 1 - self.exp_period_2_earnings_growth / self.long_term_ROE

    @property
    def years_in_period_1(self):
        return self._years_in_period_1
    
    @years_in_period_1.setter
    def years_in_period_1(self, n_years):
        if not pyfintools.tools.utils.is_integer(n_years):
            raise ValueError('The number of years in period 1 must be an integer value')
        else:
            self._years_in_period_1 = n_years
            
    @property
    def payout_ratio(self):
        """ Returns a numpy array of the payout ratio at each year in period 1, including the current period. 
        """
        if self.adjust_payout_sust:
            return np.linspace(start=self.initial_payout_ratio, 
                           stop=self.sustainable_payout_ratio, 
                           num=1 + self.years_in_period_1)
        else:
            return self.initial_payout_ratio * np.ones((1+self.years_in_period_1,), dtype=float)
            
    @property
    def earnings(self):
        if self.adjust_for_crash:
            return self._adjusted_earnings
        else:
            return self._unadjusted_earnings
        
    @property
    def _unadjusted_earnings(self):
        growth_rate = self.exp_period_1_earnings_growth
        n_years = np.arange(1 + self.years_in_period_1)
        return self.initial_earnings * np.power(1 + growth_rate, n_years)
        
    @property
    def _adjusted_earnings(self):
        unadj_earnings = self._unadjusted_earnings
    
        # Assess the loss due to a crash, based on the "earnings_drop" parameter and the unadjusted earnings
        first_year_earnings = (1 - self.earnings_drop) * unadj_earnings[0]
        
        # Determine how much of the earnings would be loss if tha earnings growth rate was not affected
        n_years = self.years_in_period_1 - 1
        multiplier = np.power(1 + self.exp_period_1_earnings_growth, n_years)
        earnings_loss = unadj_earnings[-1] - first_year_earnings * multiplier
        
        # Determine the final level of earnings, based on our assumption of what
        #   proportion of the earnings loss is recouped
        final_year_earnings = unadj_earnings[-1] - earnings_loss * (1 - self.prop_earnings_recouped)
        adj_growth_rate = -1 + np.power(final_year_earnings / first_year_earnings, 1/n_years)

        # Calculate the earnings between year 1 and the final year based on the adjusted growth rate
        DF = np.power(1 + adj_growth_rate, np.arange(n_years + 1))
        adj_earnings = np.hstack([unadj_earnings[0], first_year_earnings * DF])
        return adj_earnings

    @property
    def distributions(self):
        return self.earnings * self.payout_ratio

    @property
    def expected_terminal_value(self):
        distrib_at_start_of_period_2 = self.distributions[-1] * (1 + self.exp_period_2_earnings_growth)
        multiplier = self.current_rfr + self.long_term_ERP - self.exp_period_2_earnings_growth
        return distrib_at_start_of_period_2 / multiplier
    
    @property
    def intrinsic_value(self):
        PV_distributions = np.sum(self._present_value_of_distributions)
        PV_terminal_val = self._present_value_of_terminal_value
        return PV_distributions + PV_terminal_val

    @property
    def _present_value_of_distributions(self):
        n_years = np.arange(1 + self.years_in_period_1)
        DF = 1 / np.power(1 + self.current_rfr + self.long_term_ERP, n_years)
        return self.distributions[1:] * DF[1:]

    @property
    def _present_value_of_terminal_value(self):
        DF = 1 / np.power(1 + self.current_rfr + self.long_term_ERP, self.years_in_period_1)        
        return self.expected_terminal_value * DF

    def calculate_implied_ERP(self, reset=False, verbose=False):
        model_copy = self.copy()

        def erp_fun(erp_val):
            # For a given ERP value, Return the difference between the market value
            #   and the present value of cash flows plus the terminal value.
            model_copy.long_term_ERP = erp_val[0]
            return model_copy.market_value - model_copy.intrinsic_value

        x0 = 0.04
        result = scipy.optimize.root(erp_fun, x0)

        if verbose:
            print(result)

        if result['success']:
            erp_val = result['x'][0]
            if reset:
                self.long_term_ERP = erp_val 
            return erp_val
        else:
            raise ValueError('Unsuccessful optimization.')
