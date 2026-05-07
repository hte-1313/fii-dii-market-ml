"""
bayesian_logistic_laplace.py
----------------------------
Section 5.6 / 5.9 – Bayesian Logistic Regression with Laplace Approximation.

Posterior approximated as N(beta_hat, H^-1) around the MAP estimate.
Predicted probabilities are averaged over posterior coefficient draws.
"""

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from features import ModelMetrics


class BayesianLogisticLaplace:
    """
    Bayesian logistic classifier using a Laplace (normal) approximation
    to the posterior over coefficients.

    Parameters
    ----------
    prior_var    : float  – variance of the N(0, prior_var * I) coefficient prior
    draws        : int    – number of posterior coefficient samples for prediction
    random_state : int
    """

    def __init__(self, prior_var=1.0, draws=1000, random_state=42):
        self.prior_var = prior_var
        self.draws = draws
        self.random_state = random_state

    def fit(self, X, y):
        """
        Find MAP estimate via penalised logistic regression, then
        compute the Laplace covariance as the inverse Hessian.
        """
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        X1 = np.c_[np.ones(len(X)), X]  # prepend intercept

        # MAP estimate via L2-regularised logistic regression
        model = LogisticRegression(
            C=self.prior_var,
            fit_intercept=False,
            max_iter=3000,
            solver="lbfgs",
            class_weight="balanced",
        )
        model.fit(X1, y)
        self.coef_ = model.coef_.ravel()

        # Laplace covariance  H^{-1} = (X^T W X + prior)^{-1}
        p = expit(X1 @ self.coef_)
        w = p * (1 - p)
        prior = np.eye(X1.shape[1]) / self.prior_var
        prior[0, 0] = 1e-6  # weak prior on intercept
        H = X1.T @ (X1 * w[:, None]) + prior
        self.cov_ = np.linalg.pinv(H)
        return self

    def predict_proba(self, X):
        """
        Predictive probabilities averaged over posterior draws.

        Returns shape (n_samples, 2): columns are [P(y=0), P(y=1)].
        """
        X = np.asarray(X)
        X1 = np.c_[np.ones(len(X)), X]
        rng = np.random.default_rng(self.random_state)
        try:
            beta = rng.multivariate_normal(self.coef_, self.cov_, size=self.draws)
        except np.linalg.LinAlgError:
            beta = rng.normal(
                self.coef_,
                np.sqrt(np.diag(self.cov_)),
                size=(self.draws, len(self.coef_)),
            )
        p = expit(X1 @ beta.T).mean(axis=1)
        return np.c_[1 - p, p]


def run_bayesian_logistic_laplace(
    X_train, y_train, X_test, y_test,
    prior_var=1.0, draws=1000, random_state=42, threshold=None,
):
    """
    Convenience wrapper: fit + evaluate BayesianLogisticLaplace.

    Returns
    -------
    model      : fitted BayesianLogisticLaplace
    pred_proba : np.ndarray of P(downside=1) on test set
    metrics    : pd.Series of classification metrics
    """
    threshold = threshold if threshold is not None else y_train.mean()
    model = BayesianLogisticLaplace(
        prior_var=prior_var, draws=draws, random_state=random_state
    )
    model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = ModelMetrics.classification(y_test, pred_proba, threshold=threshold)
    return model, pred_proba, metrics
