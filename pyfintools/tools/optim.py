""" Contains optimization methods, including mean-variance, robust, and ERC

    The main function is 'optimize', and can be called to execute any of the supported
    optimization techniques.
    
    Currently supported are:
        mean-variance
        robust (Ceria-Stubbs)
        Equal Risk Contribution (ERC aka risk parity)
        robust ERC (proprietary method).
        
    Any new optimization techniques should be integrated into the 'optimize' function by
    defining a new enum in OptimMethods (in constants.py), and then extending optimize
    to call the new private function when the new optimization method is specified.
        
    This library also includes some code for working with constraints from various 
    libraries (e.g. scipy and cvxpy).

"""

import re
import numpy as np
import cvxpy
import scipy.linalg
import scipy.optimize    
from collections import defaultdict

from pyfintools.constants import OptimMethods
import pyfintools.tools.risk
import pyfintools.tools.utils


def optimize(method, **kwargs):
    if OptimMethods.MEAN_VARIANCE.value == method:
        return optimize_mean_variance(**kwargs)
    elif OptimMethods.MIN_VARIANCE.value == method:
        return optimize_min_variance(**kwargs)    
    elif OptimMethods.CVAR.value == method:
        return optimize_cvar(**kwargs)
    elif OptimMethods.ROBUST.value == method:
        return optimize_robust(**kwargs)
    elif OptimMethods.ERC.value == method:
        return optimize_erc(**kwargs)
    elif OptimMethods.ROBUST_ERC.value == method:
        return optimize_robust_erc(**kwargs)
    else:
        raise ValueError(f'Unsupported optimization method: {method}')

def _get_upper_lower_bounds(n_assets, lb=None, ub=None):
    if lb is None:
        lb = np.zeros((n_assets,), dtype=float)
    if ub is None:
        ub = np.ones((n_assets,), dtype=float)
    return lb, ub
    
def _get_constraints_cvx(asset_cov, lb=None, ub=None, unit_constraint=None, 
                         constraints=None, vol_target=None, cvx_var=None):
    """ Get the constraints in a form used by the CVXPY optimizer. """
    n_assets = asset_cov.shape[0]
    lb, ub = _get_upper_lower_bounds(n_assets, lb=lb, ub=ub)

    base_constraints = [cvx_var >= lb, cvx_var <= ub]

    if unit_constraint:
        base_constraints.append(cvx_var @ np.ones_like(lb, dtype=float) == 1)

    if vol_target is not None:
        Q2 = scipy.linalg.sqrtm(np.array(asset_cov))
        base_constraints.append(cvxpy.norm2(Q2 @ cvx_var) <= vol_target)

    if constraints is not None:
        additional_constraints = get_cvxpy_constraints(constraints, cvx_var)
        full_constraints = base_constraints + additional_constraints
    else:
        full_constraints = base_constraints
    return full_constraints

def _get_constraints_scipy(asset_cov, lb=None, ub=None, unit_constraint=None, 
                           constraints=None, vol_target=None, vol_lb=0):
    """ Get the constraints in a form used by the scipy optimizer. """
    n_assets = asset_cov.shape[0]

    # Get the lower/upper bound constraints
    lb, ub = _get_upper_lower_bounds(n_assets, lb=lb, ub=ub)
    bounds = scipy.optimize.Bounds(lb, ub)

    base_constraints = []
    if unit_constraint:
        unit_constr = scipy.optimize.LinearConstraint(np.ones((n_assets,), dtype=float), lb=1, ub=1)
        base_constraints.append(unit_constr)

    if vol_target is not None:
        var_fun = lambda x : x @ asset_cov @ x
        vol_constr = scipy.optimize.NonlinearConstraint(var_fun, lb=vol_lb ** 2, ub=vol_target ** 2)
        base_constraints.append(vol_constr)

    if constraints is not None:
        assert isinstance(constraints, list), 'The input "constraints" must be a list.'
        full_constraints = base_constraints + constraints
    else:
        full_constraints = base_constraints

    return full_constraints, bounds

def optimize_min_variance(asset_cov, lb=None, ub=None, verbose=False, unit_constraint=True, 
                          constraints=None, **kwargs):
    """ Find the minimum variance allocation, subject to constraints. """    
    asset_cov = np.array(asset_cov, dtype=float)
    n_assets = asset_cov.shape[0]

    # Handle occasional problems with solver thinking cov matrix is not pos-def
    if np.linalg.eig(asset_cov)[0].min() > -1e-10:
        asset_cov = cvxpy.psd_wrap(asset_cov)

    w = cvxpy.Variable(n_assets)
    full_constraints = _get_constraints_cvx(asset_cov, lb=lb, ub=ub, unit_constraint=unit_constraint, 
                                            constraints=constraints, cvx_var=w)
    objective = cvxpy.Minimize(cvxpy.quad_form(w, asset_cov))
    prob = cvxpy.Problem(objective=objective, constraints=full_constraints)

    fval = prob.solve(solver=cvxpy.ECOS, verbose=verbose)
    w_opt = w.value
    return w_opt, fval

def optimize_mean_variance(vol_target, mu, asset_cov, lb=None, ub=None, verbose=False, unit_constraint=True, 
                           constraints=None, **kwargs):
    return optimize_robust(vol_target=vol_target, 
                           mu=mu, 
                           asset_cov=asset_cov, 
                           unc_cov=None,
                           kappa=0.0,   # Setting kappa = 0 makes this a mean-variance optmiization
                           lb=lb, 
                           ub=ub, 
                           verbose=verbose, 
                           unit_constraint=unit_constraint,
                           constraints=constraints)

def optimize_robust(vol_target, mu, asset_cov, unc_cov, kappa, lb=None, ub=None, verbose=False, 
                    unit_constraint=True, constraints=None, **kwargs):
    # Ensure that all inputs are numpy arrays
    mu = np.array(mu, dtype=float)
    asset_cov = np.array(asset_cov, dtype=float)
    
    # Define the variable for which we will solve
    n_assets = mu.size    
    w = cvxpy.Variable(n_assets)
    full_constraints = _get_constraints_cvx(asset_cov, lb=lb, ub=ub, unit_constraint=unit_constraint, 
                                            constraints=constraints, vol_target=vol_target, cvx_var=w)
    
    if not np.isclose(kappa, 0, atol=1e-6):
        unc_cov = np.array(unc_cov, dtype=float)    
        S2 = scipy.linalg.sqrtm(np.array(unc_cov))
        objective = cvxpy.Maximize(mu @ w - kappa * cvxpy.norm2(S2 @ w))
    else:
        objective = cvxpy.Maximize(mu @ w)
    
    prob = cvxpy.Problem(objective=objective, constraints=full_constraints)

    #prob.solve(solver=cvxpy.CVXOPT, verbose=True, eps_rel=1e-6)
    fval = prob.solve(solver=cvxpy.ECOS, verbose=verbose)
    w_opt = w.value
    return w_opt, fval

def optimize_erc(asset_cov, vol_lb=0, vol_ub=np.inf, lb=None, ub=None, unit_constraint=True,
                 constraints=None, seed=None, scipy_method='SLSQP', risk_measure='var', tol=1e-10, **kwargs):
    """ Perform an Equal Risk Contribution (aka Risk Parity) optimization.

    Minimize the sum of squared deviations to equal risk from the different assets, 
    subject to the contraints.

    Arguments:
        vol_ub: (float) the minimum volatility of the allocation.
        vol_lb: (float) the maximum volatility of the allocation.
        asset_cov: (numpy array) the expected covariances of asset returns
        lb: (numpy array) the lower bounds on each allocation
        ub: (numpy array) the upper bounds on each allocation
        unit_constraint: (bool) whether to enforce that weights sum to 1 (default is True)
        constraints: (list) a list of scipy constraint objects
        tol: (float) tolerance with which constraints will be accepted
        scipy_method: (str) the scipy optimization method to be used. Default is 'SLSQP'
        seed: (int) the random seed that guarantees reproducible random numbers. If set to None,
            then the random numbers are not reproducible. Default value is None.
        """
    def _risk_budget_objective_error(weights, args):
        N = weights.size
        asset_cov = args[0]
        total_risk = pyfintools.tools.risk.calc_risk(asset_cov, weights.T,
            risk_measure=risk_measure)
        risk_contrib = pyfintools.tools.risk.calc_contribution_to_risk(asset_cov, weights.T,
            risk_measure=risk_measure)
        risk_target = total_risk/N * np.ones((N,), dtype=float)
        assert np.isclose(risk_contrib.sum(), total_risk)
        return np.sum(np.square(risk_contrib - risk_target))

    N = asset_cov.shape[0]
    full_constraints, bounds =  _get_constraints_scipy(asset_cov, lb=lb, ub=ub,
        unit_constraint=unit_constraint, constraints=constraints, vol_lb=vol_lb, vol_target=vol_ub)
    if scipy_method == 'SLSQP':
        full_constraints = get_dict_of_constraints(full_constraints)

    res = scipy.optimize.minimize(fun=_risk_budget_objective_error,
                                  x0=np.ones((N,), dtype=float) / N,
                                  args=[asset_cov],
                                  method=scipy_method,
                                  constraints=full_constraints,
                                  bounds=bounds,
                                  tol=tol,
                                  options={'disp': False})
    return res.x

def optimize_robust_erc(vol_target, mu, asset_cov, unc_cov, kappa, lb=None, ub=None, unit_constraint=True,
                        verbose=False, n_trials=500, risk_weights=1.0, constraints=None, seed=None,
                        scipy_method='SLSQP', risk_method='eig', risk_measure='entropy', **kwargs):
    """ Optmize with two objectives:
            1) maximize the expected return for a given volatility target. This primary objective
                 is maximized via the Mean-Variance optimized portfolio
            2) minimize the sum of squared contributions to risk from the different allocations.
                This objective would be maximized by an equal-risk contribution portfolio (otherwise
                known as a 'risk-parity' allocation), where each asset contributes equally to the 
                total risk of the portfolio.
        
        Arguments:
            vol_target: (float) the maximum volatility of the allocation.
            mu: (numpy array): the expected total returns (or expected excess returns)
            asset_cov: (numpy array) the expected covariances of asset returns
            unc_cov: (numpy array) the uncertainty between asset expected returns
            kappa: (float) specifies how much we can deviate from the primary objective (MV optimal expected return)
                in order to obtain our secondary objective (minimum sum or squared risk contributions)
            lb: (numpy array) the lower bounds on each allocation
            ub: (numpy array) the upper bounds on each allocation
            verbose: (int - 0, 1, 2) Verbosity levels can be passed into the optimizer to see more details on the optimization
            n_trials: (int) the optimization problem is non-convex and so results will vary based on the initial conditions.
                We can run several trials and choose the best solution from among them to ensure we are closer to the
                global optimal result.
            risk_weights: (numpy array or None) if provided, the user can specify how much weight to put on 
                each asset class when calculating the contribution to risk. By default, assets will all be equally weighted.
            constraints: (list) a list of scipy constraint objects
            scipy_method: (str) the scipy optimization method to be used. Default is 'SLSQP'
            seed: (int) the random seed that guarantees reproducible random numbers. If set to None,
                then the random numbers are not reproducible. Default value is None.
        """
    # Ensure that all inputs are numpy arrays
    mu = np.array(mu, dtype=float)
    unc_cov = np.array(unc_cov, dtype=float)
    asset_cov = np.array(asset_cov, dtype=float)

    # First, get the Mean-Variance weights
    n_assets = mu.size
    x_mv, fval_mv = optimize_mean_variance(vol_target=vol_target, mu=mu, asset_cov=asset_cov, lb=lb, ub=ub, 
                                           verbose=verbose, unit_constraint=unit_constraint, constraints=constraints)
    if x_mv is None:
        raise ValueError('Mean-variance optimization step within Robust ERC optimization has failed.')

    if np.isclose(kappa, 0.0, atol=1e-6):
        # When Kappa = 0, this optimization method is equivalent to mean-variance optimization
        return x_mv, fval_mv

    # Define the objective function (to minimize the variance of the individual contributions to risk)
    if risk_measure in ('entropy'):
        min_fun = lambda x : -pyfintools.tools.risk.calc_risk(asset_cov, x, method=risk_method, risk_measure=risk_measure)
    elif risk_measure in ('var', 'hh'):
        min_fun = lambda x : pyfintools.tools.risk.calc_risk(asset_cov, x, method=risk_method, risk_measure=risk_measure)
    else:
        raise ValueError(f'Unsupported risk measure: {risk_measure}')
    
    # Define the uncertaint constraint
    unc_fun = lambda x, x_meanvar : pyfintools.tools.risk.calc_normalized_uncertainty(mu, unc_cov, x, x_meanvar)
    unc_constr = scipy.optimize.NonlinearConstraint(lambda x : unc_fun(x, x_mv), lb=-kappa, ub=kappa)
    
    # Get the full list of constraints
    full_constraints, bounds = _get_constraints_scipy(asset_cov, lb=lb, ub=ub, unit_constraint=unit_constraint,
                                           constraints=constraints, vol_target=vol_target)
    full_constraints.append(unc_constr)
    
    # Change the class structure of the constraints, depending on the optimization method
    if scipy_method == 'SLSQP':
        full_constraints = get_dict_of_constraints(full_constraints)

    # Initialize a random state object in order to have a reproducible random number stream
    rand_state = np.random.RandomState(seed)        
        
    fval = np.inf
    w_opt = None
    tmp = []
    for _ in range(n_trials):
        x0 = rand_state.dirichlet(np.ones(n_assets))
        # Need to use the 'SLSQP' solver, as it is the only one that currently respects constraints
        out = scipy.optimize.minimize(min_fun, x0, method=scipy_method, bounds=bounds, constraints=full_constraints)
        if out['success']:
            tmp.append(out['x'])
            val = min_fun(out['x'])
            if val < fval:
                fval = val
                w_opt = out['x']
    if w_opt is None:
        raise ValueError('Unsuccessful')
    return w_opt, fval

def optimize_cvar(vol_target, returns, confidence, lb=None, ub=None, unit_constraint=True, constraints=None, **kwargs):
    raise NotImplementedError('Need to implement CVaR optimization.')

def get_cvxpy_constraints(constraints, w):
    scipy_constraint_types = (scipy.optimize.LinearConstraint, scipy.optimize.NonlinearConstraint)
    if np.all([isinstance(c, scipy_constraint_types) for c in constraints]):
        constraint_dict = get_dict_of_constraints(constraints)
    elif np.all([isinstance(c, dict) for c in constraints]):
        constraint_dict = constraints
    else:
        raise ValueError('Unsupported constraint input. Entries must all be scipy constraints or dict.')
        
    additional_constraints = []
    if constraint_dict is not None:
        for _constraint in constraint_dict:
            cfun = _constraint['fun']
            ctype = _constraint['type']
            if ctype == 'eq':
                additional_constraints.append(cfun(w) == 0)
            elif ctype == 'ineq':
                additional_constraints.append(cfun(w) >= 0)
            else:
                raise ValueError(f'Unknown constraint type: {ctype}')
    return additional_constraints

def _remove_parenthesis(expression):
    """ Remove parenthesis from a constraint equation and change coefficients as necessary. """
    # Not implemented yet...
    return expression

def _is_numeric_string(expression):
    """ Return True/False if an expression represents a numeric string quantity (e.g. '1.2', '23')"""
    return 0 == len(re.sub('[0-9\.\-]', '', expression))

def _parse_coefficient(components, operators):
    """ Extract a floating number from a string coefficient (e.g. '-1/4' --> -0.25) """
    value = None
    for j, cmp in enumerate(components):
        if not _is_numeric_string(cmp):
            raise ValueError('Parsing is only supported for non-symbolic expressions.')
        else:
            new_val = eval(cmp)
            if value is None:
                value = new_val
            else:
                op = operators[j-1]
                if op == '*':
                    value *= new_val
                elif op == '/':
                    value /= new_val
                else:
                    raise ValueError(f'Unsupported operator: {op}')
    return value

def _extract_single_symbol_and_coefficient(components, operators):
    """ Extract the symbol and coefficient of an atomic expression.
        e.g.  '12 * SPY' --return--> ('SPY', 12.0)  
    """
    codes = [cmp for cmp in components if re.search('[a-zA-Z]', cmp)]

    CONSTANT = None
    if len(codes) == 0:
        symbol = None
        value = _parse_coefficient(components, operators)
    elif len(codes) == 1:
        symbol = codes[0]
        idx = components.index(symbol)
        if idx > 0 and operators[idx-1] != '*':
            raise ValueError('Asset symbols can only be multiplied by coefficients.')
        else:
            components[idx] = '1'
            value = _parse_coefficient(components, operators)
    else:
        raise ValueError('Only linear constraints are supported.')
    return symbol, value

def _parse_one_side_of_constraint_equation(eq_side):
    """ Parse one side of the constraint equation. Return a dictionary with coefficients for each symbol.
        For constant values, the dictionary key is None. """
    values = defaultdict(int)
    eq_side = re.sub('-', '+-', eq_side)
    expressions = re.split('\+', eq_side)

    for expression in expressions:
        if expression:
            components = re.split('[\*\/]', expression)
            operators = list(re.sub('[a-zA-Z0-9\.\-]', '', expression))
            symbol, val = _extract_single_symbol_and_coefficient(components, operators)
            values[symbol] += val
    return values


def _find_subexpressions(string):
    """ Find start and end index for sub-expressions contained within parentheses. """
    if '(' in string:
        idx_start = re.search('\(', string).start()
        ct = 1    
        for idx_end in range(idx_start+1, len(string)):
            if string[idx_end] == '(':
                ct += 1
            elif string[idx_end] == ')':
                ct -= 1

            if ct == 0:
                break
        return idx_start, idx_end
    else:
        return None, None

def _remove_parentheses(string):
    """ Remove parentheses from a one-sided expression. Returns the modified expression, containing 
        random temporary placeholders for sub-expressions contained within parentheses.
        Also returns a placeholder dictionary containing information about the sub-expressions. 
        """
    placeholders = dict()

    idx_start, idx_end = _find_subexpressions(string)
    while idx_start is not None:
        # Create a temporary random symbol as a placeholder for the expression
        expression = string[idx_start+1:idx_end]        
        random_symbol = pyfintools.tools.utils.generate_random_string(10)
        placeholders[random_symbol] = expression

        # Replace the expression with the temporary random symbol
        string = ''.join([string[:idx_start], random_symbol, string[idx_end+1:]])
        
        # Recursively replace any parentheses in the expression as well
        new_expression, plc = _remove_parentheses(expression)
        placeholders[random_symbol] = new_expression
        placeholders.update(plc)

        idx_start, idx_end = _find_subexpressions(string)

    return string, placeholders

def _parse_one_side_of_constraint_equation_without_parentheses(eq_side):
    """ Parse one side of the constraint equation. Return a dictionary with coefficients for each symbol.
        For constant values, the dictionary key is None. Expression may not contain parentheses."""    
    values = defaultdict(int)
    eq_side = re.sub('-', '+-', eq_side)
    expressions = re.split('\+', eq_side)

    for expression in expressions:
        if expression:
            components = re.split('[\*\/]', expression)
            operators = list(re.sub('[a-zA-Z0-9\.\-]', '', expression))
            symbol, val = _extract_single_symbol_and_coefficient(components, operators)
            values[symbol] += val
    return values    

def _parse_one_side_of_constraint_equation(eq_side):
    """ Parse one side of the constraint equation. Return a dictionary with coefficients for each symbol.
        For constant values, the dictionary key is None. Expression may contain parentheses. """
    expression, placeholders = _remove_parentheses(eq_side)
    output = _parse_one_side_of_constraint_equation_without_parentheses(expression)

    finished = False
    while not finished:
        finished = True
        for symbol, coef in output.items():
            if symbol in placeholders:
                sub_expr = placeholders[symbol]
                sub_output = _parse_one_side_of_constraint_equation_without_parentheses(sub_expr)
                for sub_symbol, sub_coef in sub_output.items():
                    output[sub_symbol] = coef * sub_coef
                del output[symbol]
                finished = False
                break
    return output

def _parse_constraint_equation(constraint):
    """ Effectively move all of the symbols to the left-hand side of the constraint equation, 
        and move any constant term to the righ-hand side.
        Return a dictionary of coefficients for the resulting equation. The dictionary
        keys are the symbols, and the constant term on the right-hand side has a key of None. """
    # Remove any whitespace
    constraint = constraint.replace(' ', '')

    # Make sure we don't have adjacent '+' and '-', to simplify parsing
    while '+-' in constraint or '-+' in constraint:
        constraint = constraint.replace('-+', '-')
        constraint = constraint.replace('+-', '-')    

    # Split the equation into left-hand and right-hand sides
    sides = []
    for comparator in ['<=', '>=', '=', '<', '>']:
        if not sides and comparator in constraint:
            sides = re.split(comparator, constraint)
            break

    if len(sides) == 2:
        lhs, rhs = sides
    else:
        raise ValueError('Constraint expression should have exactly one comparator (=, <, >, <=, <=).')
        
    # Parse the left- and right-hand sides
    lhs_values = _parse_one_side_of_constraint_equation(lhs)
    rhs_values = _parse_one_side_of_constraint_equation(rhs)

    # Combine the left- and right-hand sides into a single expression
    for k, v in rhs_values.items():
        lhs_values[k] -= v
    if None not in lhs_values:
        lhs_values[None] = 0.0
    else:
        lhs_values[None] *= -1  # Move the constant term to the right-hand side
    return lhs_values, comparator

def _get_scipy_constraint_from_string_single(constraint_string, asset_codes):
    """ Parse a string constraint and convert it into a scipy Linear constraint. 
    
        Arguments:
            constraint_string: (str) is the string constraint, whose form is described below.
            asset_codes: (numpy array) is a list of string asset codes to which the constraint
                    string makes reference. Asset codes cannot contain whitespace, and must contain
                    at least one alphabetical character.

        Examples of supported constraint_string:
            "EMCORP = 1/3 * EMSOV"
            "2 * USEQ + 5 * EUREQ + 10 * JPEQ = 0.40"        
    """
    if isinstance(asset_codes, str):
        asset_codes = np.array([asset_codes])
    else:
        asset_codes = np.array(list(asset_codes))
        
    assert ';' not in constraint_string, 'Single constraints may not contain the symbol ";"'
    parsed, comparator = _parse_constraint_equation(constraint_string)
    A = np.zeros((asset_codes.size,), dtype=float)
    b = parsed[None]
    for code, coef in parsed.items():
        if code is not None:
            idx = np.where(asset_codes == code)[0][0]
            A[idx] = coef

    if comparator == '=':
        lb = ub = b
    elif comparator in ('<', '<='):
        lb = -np.inf
        ub = b
    elif comparator in ('>', '>='):
        lb = b
        ub = np.inf
    else:
        raise ValueError(f'Unknown comparator: {comparator}')

    return scipy.optimize.LinearConstraint(A, lb=lb, ub=ub)

def get_scipy_constraints_from_string(constraint_string, asset_codes):
    """ Parse a string constraint and convert it into a list of scipy Linear constraints. 
        Individual constraints within the constraint_string should be separated by ";"
    
        Arguments:
            constraint_string: (str) is the string constraint, whose form is described below.
            asset_codes: (numpy array) is a list of string asset codes to which the constraint
                    string makes reference. Asset codes cannot contain whitespace, and must contain
                    at least one alphabetical character.

        Examples of supported constraint_string:
            "EMCORP = 1/3 * EMSOV;DMEQ = 9*EMEQ"
            "2 * USEQ + 5 * EUREQ + 10 * JPEQ = 0.40"        
    """
    single_constraints = re.split(';', constraint_string)
    scipy_constraints = []
    for constraint in single_constraints:
        if constraint:
            _constr = _get_scipy_constraint_from_string_single(constraint, asset_codes)
            scipy_constraints.append(_constr)
    return scipy_constraints    


def _constraint_helper_fun(_constraint):
    if isinstance(_constraint, scipy.optimize.LinearConstraint):
        return lambda x : _constraint.A @ x
    elif isinstance(_constraint, scipy.optimize.NonlinearConstraint):
        return lambda x : _constraint.fun(x)
    else:
        raise ValueError('Unsupported constraint type: {}'.format(_constraint.__class__))

def _create_lb_constraint_dict(_constraint, _type):
    local_fun = _constraint_helper_fun(_constraint)
    if -np.inf < _constraint.lb:
        lb_fun = lambda x : local_fun(x) - _constraint.lb
        return {'type' : _type, 'fun' : lb_fun}
    else:
        return None    

def _create_ub_constraint_dict(_constraint, _type):
    local_fun = _constraint_helper_fun(_constraint)
    if _constraint.ub < np.inf and _type != 'eq':
        ub_fun = lambda x : _constraint.ub - local_fun(x)
        return {'type' : _type, 'fun' : ub_fun}
    else:
        return None    

def get_dict_of_constraints(constraints, rtol=1e-05, atol=1e-08):
    constraint_list = []
    for _constraint in constraints:
        if np.isclose(_constraint.lb, _constraint.ub, atol=atol, rtol=rtol):
            _type = 'eq'
        else:
            _type = 'ineq'

        lb_constraint = _create_lb_constraint_dict(_constraint, _type)
        if lb_constraint is not None:
            constraint_list.append(lb_constraint)
            
        ub_constraint = _create_ub_constraint_dict(_constraint, _type)
        if ub_constraint is not None:
            constraint_list.append(ub_constraint)

    return constraint_list