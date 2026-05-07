"""
logistic.py
-----------
Section 5.3 – Baseline downside classifier: Logistic Regression.

Fits a logistic regression model with class_weight='balanced' on the
base feature set and evaluates probabilistic classification metrics.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from features import ModelMetrics


def run_logistic(X_train, y_train, X_test, y_test, threshold=0.5, max_iter=3000):
    """
    Fit logistic regression and return (model, predicted_proba, metrics).

    Parameters
    ----------
    X_train, X_test : np.ndarray
    y_train, y_test : np.ndarray  – binary downside labels
    threshold       : float        – classification threshold (default 0.5)
    max_iter        : int

    Returns
    -------
    model       : fitted LogisticRegression
    pred_proba  : np.ndarray of P(downside=1) on test set
    metrics     : pd.Series of classification metrics
    """
    model = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
    ).fit(X_train, y_train)

    pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = ModelMetrics.classification(y_test, pred_proba, threshold=threshold)

    # Warn if classifier is degenerate
    recall_val = metrics["Recall"]
    if recall_val > 0.95:
        print(
            f"WARNING: Recall = {recall_val:.3f} — classifier may be predicting "
            "all positives. Check threshold."
        )
    else:
        print(f"Recall = {recall_val:.3f} — classifier is discriminating.")

    return model, pred_proba, metrics
