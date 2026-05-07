"""
bayesian_ridge.py
-----------------
Section 5.5 – Bayesian Ridge return regression.

Extends the OLS benchmark by placing a prior on regression coefficients
and returning predictive uncertainty intervals alongside point forecasts.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

from features import ModelMetrics


def run_bayesian_ridge(X_train, r_train, X_test, r_test, plot=True):
    """
    Fit Bayesian Ridge and return (model, predictions, std_devs, metrics).

    Parameters
    ----------
    X_train, X_test : np.ndarray
    r_train, r_test : np.ndarray  – realised 5-day forward returns
    plot            : bool        – show predictive interval plot

    Returns
    -------
    model   : fitted BayesianRidge
    pred    : np.ndarray of point predictions on test set
    std     : np.ndarray of predictive standard deviations on test set
    metrics : pd.Series of regression metrics
    """
    model = BayesianRidge().fit(X_train, r_train)
    pred, std = model.predict(X_test, return_std=True)
    metrics = ModelMetrics.regression(r_test, pred)

    if plot:
        n = min(150, len(r_test))
        plt.figure(figsize=(10, 4))
        plt.plot(r_test[:n], label="actual")
        plt.plot(pred[:n], label="predicted")
        plt.fill_between(
            np.arange(n),
            pred[:n] - 1.64 * std[:n],
            pred[:n] + 1.64 * std[:n],
            alpha=0.2,
        )
        plt.title("Bayesian Ridge predictive interval (90%)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return model, pred, std, metrics
