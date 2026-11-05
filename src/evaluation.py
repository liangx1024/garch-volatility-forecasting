"""Variance forecast evaluation."""

import numpy as np


def forecast_scores(forecast, proxy):
    """Compare positive variance forecasts and squared shocks on matching dates."""
    
    return {
        "QLIKE": (np.log(forecast) + proxy / forecast).mean(),
        "MSE": ((proxy - forecast) ** 2).mean(),
    }
