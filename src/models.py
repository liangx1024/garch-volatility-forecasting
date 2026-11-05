"""One-step-ahead daily variance forecasts."""

import numpy as np
import pandas as pd


def rolling_variance(returns, L):
    """Forecast daily variance; assumes 1 < L < len(returns)."""

    # we can call the built-in function as follows
    # return returns.rolling(window=L).var(ddof=1).shift(1)
    # but we use our own code to explicitly demonstrate the computing process

    r = returns.to_numpy()
    h = np.full(len(r), np.nan)

    # Each row contains L past returns for one forecast date.
    windows = np.lib.stride_tricks.sliding_window_view(r[:-1], L)
    means = np.sum(windows, axis=1, keepdims=True) / L
    h[L:] = np.sum((windows - means) ** 2, axis=1) / (L - 1)

    return pd.Series(h, index=returns.index)


def ewma_variance(returns, lam, h0, mean=0.0):
    """Forecast daily variance using prior shocks and starting variance h0."""

    # we can use the built-in .ewm method
    # but for explicit demonstration we use the following code

    squared_shocks = (returns.to_numpy() - mean) ** 2
    h = np.full(len(returns), np.nan)
    h[0] = h0

    for i in range(1, len(returns)):
        h[i] = lam * h[i - 1] + (1 - lam) * squared_shocks[i - 1]

    return pd.Series(h, index=returns.index)


def garch_variance(returns, omega, alpha, beta, h0, mean=0.0):
    """Forecast daily variance using the GARCH(1,1) recursion."""

    # there is an established package, arch, for GARCH implementation
    # but for explicit demonstration we use our own code
    
    squared_shocks = (returns.to_numpy() - mean) ** 2
    h = np.full(len(returns), np.nan)
    h[0] = h0

    for i in range(1, len(returns)):
        h[i] = omega + alpha * squared_shocks[i - 1] + beta * h[i - 1]

    return pd.Series(h, index=returns.index)
