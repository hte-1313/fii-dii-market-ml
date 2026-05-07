"""
xgboost_model.py
----------------
Section 5.4 – Nonlinear downside classifier: XGBoost.

Fits an XGBoost classifier on the full feature set (base + game variables)
with class-imbalance weighting.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from features import ModelMetrics


def run_xgboost(
    X_train,
    y_train,
    X_test,
    y_test,
    random_state=42,
    n_estimators=250,
    max_depth=3,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    threshold=None,
):
    """
    Fit XGBoost with scale_pos_weight for class imbalance.

    Parameters
    ----------
    X_train, X_test   : np.ndarray
    y_train, y_test   : np.ndarray  – binary downside labels
    threshold         : float or None; defaults to train event rate if None

    Returns
    -------
    model      : fitted XGBClassifier
    pred_proba : np.ndarray of P(downside=1) on test set
    metrics    : pd.Series of classification metrics
    """
    spw = max(1, sum(y_train == 0) / max(1, sum(y_train == 1)))
    threshold = threshold if threshold is not None else y_train.mean()

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = ModelMetrics.classification(y_test, pred_proba, threshold=threshold)

    return model, pred_proba, metrics
