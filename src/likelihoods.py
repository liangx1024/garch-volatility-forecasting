"""Gaussian and variance-one Student's t GARCH likelihoods."""

import numpy as np
from scipy.special import gammaln

from .models import garch_variance


def garch_nll(params, returns, h0, mean):
    """Gaussian negative log-likelihood with fixed mean and initial variance."""
    
    omega, alpha, beta = params
    h = garch_variance(returns, omega, alpha, beta, h0, mean).to_numpy()
    squared_shocks = (returns.to_numpy() - mean) ** 2
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + squared_shocks / h)


def garch_t_nll(params, returns, h0, mean):
    """Negative log-likelihood for variance-one Student's t shocks."""
    
    omega, alpha, beta, nu = params
    h = garch_variance(returns, omega, alpha, beta, h0, mean).to_numpy()
    squared_shocks = (returns.to_numpy() - mean) ** 2
    constant = gammaln(nu / 2) - gammaln((nu + 1) / 2)
    return np.sum(
        constant + 0.5 * np.log(np.pi * (nu - 2) * h)
        + 0.5 * (nu + 1) * np.log1p(squared_shocks / ((nu - 2) * h))
    )
