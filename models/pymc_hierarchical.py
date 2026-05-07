"""
pymc_hierarchical.py
--------------------
Section 5.9 (advanced) – Hierarchical Bayesian logistic regression via PyMC.

Model specification
-------------------
  y_{s,t} ~ Bernoulli(sigmoid(alpha_s + X_{s,t} @ beta))

  alpha_s  ~ N(mu_alpha, sigma_alpha^2)   [partial pooling across sectors]
  mu_alpha ~ N(0, 1)
  sigma_alpha ~ HalfNormal(1)

  beta_j   ~ N(0, tau^2 * lambda_j^2)    [local-global shrinkage]
  tau      ~ HalfCauchy(1)
  lambda_j ~ HalfCauchy(1)

Posterior inference via NUTS (PyMC).
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

from config import Config
from features import ModelMetrics


class PyMCHierarchicalGameModel:
    """
    Hierarchical Bayesian logistic regression with sector-level partial pooling
    and horseshoe-like coefficient shrinkage, fitted via PyMC NUTS sampling.

    Parameters
    ----------
    cfg : Config
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scaler = StandardScaler()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_rows(self, train):
        """Down-sample training data if it exceeds pymc_max_rows."""
        if len(train) <= self.cfg.pymc_max_rows:
            return train.copy()
        parts = []
        per_class = max(300, self.cfg.pymc_max_rows // 2)
        for _, g in train.groupby("downside"):
            parts.append(
                g.sample(min(len(g), per_class), random_state=self.cfg.random_state)
            )
        return (
            pd.concat(parts)
            .sample(frac=1, random_state=self.cfg.random_state)
            .head(self.cfg.pymc_max_rows)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train, feature_cols):
        """
        Fit the hierarchical model on training panel data.

        Parameters
        ----------
        train        : pd.DataFrame  – must contain 'sector' and 'downside' columns
        feature_cols : list[str]     – predictor columns

        Returns
        -------
        self
        """
        tr = self._sample_rows(train)
        self.feature_cols = list(feature_cols)
        self.sectors = sorted(train["sector"].dropna().unique())

        X_raw = tr[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        self.scaler.fit(X_raw)
        X = self.scaler.transform(X_raw).astype("float64")
        y = tr["downside"].astype(int).values
        sector_idx = pd.Categorical(
            tr["sector"], categories=self.sectors
        ).codes.astype("int64")

        coords = {"feature": self.feature_cols, "sector": self.sectors}

        n_used = len(tr)
        n_full = len(train)
        print(
            f"PyMC model fitted on {n_used:,} of {n_full:,} training rows "
            f"({100 * n_used / n_full:.1f}%)."
        )

        with pm.Model(coords=coords) as model:
            X_data = pm.Data("X_data", X)
            sector_data = pm.Data("sector_data", sector_idx)

            # Sector intercepts (hierarchical)
            mu_alpha = pm.Normal("mu_alpha", 0.0, 1.0)
            sigma_alpha = pm.HalfNormal("sigma_alpha", 1.0)
            alpha_raw = pm.Normal("alpha_raw", 0.0, 1.0, dims="sector")
            alpha = pm.Deterministic(
                "alpha", mu_alpha + sigma_alpha * alpha_raw, dims="sector"
            )

            # Coefficients with local-global shrinkage
            tau = pm.HalfCauchy("tau", beta=1.0)
            lam = pm.HalfCauchy("lambda", beta=1.0, dims="feature")
            beta = pm.Normal("beta", 0.0, tau * lam, dims="feature")

            # Likelihood
            eta = alpha[sector_data] + pm.math.dot(X_data, beta)
            pm.Bernoulli("y", logit_p=eta, observed=y)

            idata = pm.sample(
                draws=getattr(self.cfg, "pymc_draws", 800),
                tune=getattr(self.cfg, "pymc_tune", 600),
                chains=getattr(self.cfg, "pymc_chains", 2),
                target_accept=0.90,
                init="adapt_diag",
                nuts_sampler_kwargs={"max_treedepth": 15},
                random_seed=self.cfg.random_state,
                progressbar=True,
            )

        divs = int(idata.sample_stats["diverging"].values.sum())
        if divs > 0:
            print(f"WARNING: {divs} divergences remain.")
        else:
            print("No divergences — sampling looks healthy.")

        self.model = model
        self.idata = idata
        return self

    def predict_proba(self, df):
        """
        Posterior predictive probabilities for each row in `df`.

        Returns
        -------
        np.ndarray of shape (n,) — P(downside=1) averaged over posterior draws.
        """
        X_raw = df[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        X = self.scaler.transform(X_raw).astype("float64")
        sector_idx = pd.Categorical(df["sector"], categories=self.sectors).codes

        posterior = self.idata.posterior
        beta = (
            posterior["beta"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "feature")
            .values
        )
        alpha = (
            posterior["alpha"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", "sector")
            .values
        )

        p = np.full(len(df), np.nan)
        valid = sector_idx >= 0
        eta = X[valid] @ beta.T + alpha[:, sector_idx[valid]].T
        p[valid] = expit(eta).mean(axis=1)
        return p

    def coefficient_summary(self):
        """
        Return a DataFrame of posterior beta coefficient summaries
        (mean, sd, HDI, r_hat, ESS) sorted by |mean|.
        """
        beta_summary = az.summary(
            self.idata, var_names=["beta"], round_to=3
        )
        beta_summary["feature"] = self.feature_cols
        return (
            beta_summary[
                ["feature", "mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk"]
            ]
            .sort_values("mean", key=np.abs, ascending=False)
        )

    def sampling_diagnostics(self):
        """Return a one-row DataFrame of key MCMC diagnostics."""
        summary = az.summary(self.idata, round_to=3)
        out = {
            "max_rhat": summary["r_hat"].max() if "r_hat" in summary.columns else np.nan,
            "min_ess_bulk": (
                summary["ess_bulk"].min() if "ess_bulk" in summary.columns else np.nan
            ),
            "min_ess_tail": (
                summary["ess_tail"].min() if "ess_tail" in summary.columns else np.nan
            ),
            "divergences": (
                int(self.idata.sample_stats["diverging"].sum().values)
                if "diverging" in self.idata.sample_stats
                else np.nan
            ),
            "mean_acceptance_rate": (
                float(self.idata.sample_stats["acceptance_rate"].mean().values)
                if "acceptance_rate" in self.idata.sample_stats
                else np.nan
            ),
        }
        return pd.DataFrame([out])
