"""
ensemble_benchmarks.py
----------------------
Section 6 – Nonlinear benchmark classifiers for fair model comparison.

Models
------
- XGBoost (full features)
- Random Forest          (calibrated, isotonic)
- Extra Trees            (calibrated, isotonic)
- Hist Gradient Boosting (calibrated, isotonic)
- Gaussian Process       (only when training set is small enough)
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, RBF
from xgboost import XGBClassifier

from features import ModelMetrics


def run_random_forest(
    X_train, y_train, X_test, y_test,
    random_state=42, n_estimators=300, max_depth=6, threshold=None,
):
    """Calibrated Random Forest classifier."""
    threshold = threshold if threshold is not None else y_train.mean()
    base = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = ModelMetrics.classification(y_test, pred_proba, threshold=threshold)
    return model, pred_proba, metrics


def run_extra_trees(
    X_train, y_train, X_test, y_test,
    random_state=42, n_estimators=400, max_depth=8, threshold=None,
):
    """Calibrated Extra Trees classifier."""
    threshold = threshold if threshold is not None else y_train.mean()
    base = ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = ModelMetrics.classification(y_test, pred_proba, threshold=threshold)
    return model, pred_proba, metrics


def run_hist_gradient_boosting(
    X_train, y_train, X_test, y_test,
    random_state=42, max_iter=250, learning_rate=0.04,
    max_leaf_nodes=15, threshold=None,
):
    """Calibrated Histogram Gradient Boosting classifier."""
    threshold = threshold if threshold is not None else y_train.mean()
    base = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        random_state=random_state,
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = ModelMetrics.classification(y_test, pred_proba, threshold=threshold)
    return model, pred_proba, metrics


def run_gaussian_process(
    X_train, y_train, X_test, y_test,
    random_state=42, max_train_rows=4000, threshold=None,
):
    """
    Gaussian Process classifier — skipped (returns NaN metrics) when
    training set exceeds max_train_rows due to O(n^3) cost.
    """
    threshold = threshold if threshold is not None else y_train.mean()
    nan_metrics = pd.Series(
        {k: np.nan for k in ["LogLoss", "Brier", "Accuracy", "Precision",
                              "Recall", "F1", "AUC", "PR_AUC",
                              "EventRate", "TopDecileLift"]}
    )
    if len(X_train) > max_train_rows:
        print(
            f"Gaussian Process skipped: training set ({len(X_train):,} rows) "
            f"exceeds max_train_rows={max_train_rows}."
        )
        return None, None, nan_metrics

    model = GaussianProcessClassifier(
        kernel=RBF() + DotProduct(), random_state=random_state
    )
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = ModelMetrics.classification(y_test, pred_proba, threshold=threshold)
    return model, pred_proba, metrics


def run_all_benchmarks(
    X_train, y_train, X_test, y_test,
    random_state=42, threshold=None,
):
    """
    Fit all ensemble benchmark models and return a comparison DataFrame.

    Returns
    -------
    results   : dict  – {name: (model, pred_proba, metrics)}
    table     : pd.DataFrame – metrics for each model, sorted by LogLoss
    """
    threshold = threshold if threshold is not None else y_train.mean()
    kwargs = dict(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        random_state=random_state, threshold=threshold,
    )

    results = {
        "Random Forest (calibrated)": run_random_forest(**kwargs),
        "Extra Trees (calibrated)": run_extra_trees(**kwargs),
        "Hist Gradient Boosting (calibrated)": run_hist_gradient_boosting(**kwargs),
        "Gaussian Process": run_gaussian_process(**kwargs),
    }

    table = pd.DataFrame(
        {name: r[2] for name, r in results.items()}
    ).T.sort_values("LogLoss")

    return results, table
