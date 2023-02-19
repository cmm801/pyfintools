""" A package containing some common risk functions.
"""

import numpy as np


def calc_uncertainty_distance(unc_cov, x_0, x_1):
    return np.sqrt((x_0 - x_1) @ unc_cov @ (x_0 - x_1)) + 1e-10

def calc_normalized_uncertainty(mu, unc_cov, x_0, x_1):
    unc_dist = calc_uncertainty_distance(unc_cov, x_0, x_1)
    return mu @ (x_0 - x_1) / unc_dist

def calc_contribution_to_risk(asset_cov, w, risk_measure='var', method='asset'):
    if risk_measure == 'var':
        return calc_contribution_to_var(asset_cov, w, method=method)
    elif risk_measure == 'hh':
        return calc_contribution_to_hh_index(asset_cov, w, method=method)
    elif risk_measure == 'entropy':
        return calc_contribution_to_entropy(asset_cov, w, method=method)
    else:
        raise ValueError(f'Unknown risk measure: {risk_measure}')

def calc_contribution_to_var(asset_cov, w, method='asset'):
    if method == 'asset':
        return calc_contribution_to_var_by_asset(asset_cov, w)
    elif method == 'eig':
        return calc_contribution_to_var_by_eig(asset_cov, w)
    else:
        raise ValueError(f'Unsupported method: {method}')

def calc_contribution_to_var_by_asset(asset_cov, w):
    contrib_to_var = w * (asset_cov @ w)
    return contrib_to_var

def calc_contribution_to_var_by_eig(asset_cov, w):
    eig_vals, eig_vectors = scipy.linalg.eig(asset_cov)
    contrib_to_var = np.power(eig_vectors.T @ w, 2) * eig_vals    
    if not np.all(np.isclose(0, contrib_to_var.imag)):
        raise ValueError('Eigenvalues should not have non-zero imaginary part.')
    else:
        contrib_to_var = contrib_to_var.real
    return contrib_to_var

def calc_contribution_to_hh_index(asset_cov, w, method='asset'):
    contrib_to_var = calc_contribution_to_var(asset_cov, w, method=method)
    return np.power(contrib_to_var, 2)

def calc_contribution_to_entropy(asset_cov, w, method='asset'):
    p = calc_contribution_to_var(asset_cov, w, method=method)
    
    # Do not allow negative 'probabilities', since the Log function is undefined for p <= 0
    p = np.maximum(p, 1e-10)
    
    # Normalize so distribution sums to 1
    p /= np.sum(p)

    # Calculate the contribution to informational entropy
    return -p * np.log(p)

def calc_risk(asset_cov, w, risk_measure='var', method='asset'):
    return np.sum(calc_contribution_to_risk(asset_cov, w, risk_measure=risk_measure, method=method))
    
def calc_risk_entropy(asset_cov, w, method='asset'):
    return calc_risk(asset_cov, w, risk_measure='entropy', method=method)

def calc_risk_hh_index(asset_cov, w, method='asset'):
    return calc_risk(asset_cov, w, risk_measure='hh', method=method)
