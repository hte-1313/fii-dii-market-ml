"""
ols_ridge_lasso.py
------------------
Section 5.1 / 5.2 – Baseline return regression models.

Models
------
- OLS  (LinearRegression)
- Ridge
- Lasso
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from features import ModelMetrics


def run_ols(X_train, y_train, X_test, y_test):
    """Fit OLS and return (model, predictions, metrics)."""
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = ModelMetrics.regression(y_test, pred)
    return model, pred, metrics


def run_ridge(X_train, y_train, X_test, y_test, alpha=1.0):
    """Fit Ridge regression and return (model, predictions, metrics)."""
    model = Ridge(alpha=alpha).fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = ModelMetrics.regression(y_test, pred)
    return model, pred, metrics


def run_lasso(X_train, y_train, X_test, y_test, alpha=0.0005, max_iter=20000):
    """Fit Lasso regression and return (model, predictions, metrics)."""
    model = Lasso(alpha=alpha, max_iter=max_iter).fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = ModelMetrics.regression(y_test, pred)
    return model, pred, metrics


def compare_regression_models(X_train, r_train, X_test, r_test):
    """
    Fit OLS, Ridge and Lasso on the same data and return a comparison table.

    Parameters
    ----------
    X_train, X_test : np.ndarray  – feature matrices
    r_train, r_test : np.ndarray  – realised 5-day forward returns

    Returns
    -------
    pd.DataFrame with one row per model and regression metric columns.
    """
    _, _, ols_m = run_ols(X_train, r_train, X_test, r_test)
    _, _, ridge_m = run_ridge(X_train, r_train, X_test, r_test)
    _, _, lasso_m = run_lasso(X_train, r_train, X_test, r_test)

    return pd.DataFrame(
        [ols_m, ridge_m, lasso_m],
        index=["OLS", "Ridge", "Lasso"],
    )
