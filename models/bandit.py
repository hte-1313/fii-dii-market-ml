"""
bandit.py
---------
Section 7 – Bayesian contextual bandit allocation layer.

BayesianLinearBandit  – Thompson-sampling linear bandit with Bayesian updates.
make_bandit_frame     – builds reward table from panel predictions.
run_bandit            – warm-up on training period, evaluate on test period.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import Config


# ---------------------------------------------------------------------------
# Bandit model
# ---------------------------------------------------------------------------

class BayesianLinearBandit:
    """
    Linear Thompson-sampling contextual bandit.

    Maintains a Bayesian posterior over the reward-weight vector theta_a
    for each action a.  At each step:
      1. Sample theta_a ~ N(m_a, noise_var * V_a) for every action.
      2. Choose the action with the highest sampled expected reward x @ theta_a.
      3. Update posterior with observed (x, r).

    Parameters
    ----------
    actions         : list  – available allocation actions
    d               : int   – context dimension (including intercept)
    noise_var       : float – assumed reward noise variance
    prior_precision : float – precision of N(0, I/prior_precision) prior
    random_state    : int
    """

    def __init__(self, actions, d, noise_var=0.05, prior_precision=1.0, random_state=42):
        self.actions = list(actions)
        self.d = d
        self.noise_var = noise_var
        self.rng = np.random.default_rng(random_state)
        self.A = {a: np.eye(d) * prior_precision for a in self.actions}
        self.b = {a: np.zeros(d) for a in self.actions}

    def posterior(self, a):
        """Return posterior (mean, covariance) for action a."""
        V = np.linalg.inv(self.A[a])
        m = V @ self.b[a]
        return m, V

    def choose(self, x):
        """Thompson-sample and return the best action for context x."""
        scores = {}
        for a in self.actions:
            m, V = self.posterior(a)
            theta = self.rng.multivariate_normal(m, self.noise_var * V)
            scores[a] = x @ theta
        return max(scores, key=scores.get)

    def update(self, a, x, r):
        """Bayesian update after observing reward r for action a in context x."""
        self.A[a] += np.outer(x, x) / self.noise_var
        self.b[a] += x * r / self.noise_var


# ---------------------------------------------------------------------------
# Helper: build reward table from panel
# ---------------------------------------------------------------------------

def make_bandit_frame(panel, daily, pred_col, cfg: Config):
    """
    Build a per-date reward table by aggregating panel predictions.

    Columns in the output
    ---------------------
    date, mean_downside_prob, max_downside_prob,
    cash, nifty, defensive, cyclical, low_risk
    (plus all daily columns for context)

    Parameters
    ----------
    panel    : pd.DataFrame  – sector panel with prediction column
    daily    : pd.DataFrame  – daily market/flow/regime table
    pred_col : str           – column name with predicted downside probability
    cfg      : Config

    Returns
    -------
    pd.DataFrame sorted by date
    """
    daily_map = daily.set_index("date")
    rows = []
    for d, g in panel.groupby("date"):
        if d not in daily_map.index:
            continue
        if pred_col not in g.columns:
            continue
        row = daily_map.loc[d].to_dict()
        low = g.nsmallest(min(3, len(g)), pred_col)
        defensive = g[g["sector"].isin(cfg.defensive_sectors)]
        cyclical = g[g["sector"].isin(cfg.cyclical_sectors)]

        row["date"] = d
        row["mean_downside_prob"] = g[pred_col].mean()
        row["max_downside_prob"] = g[pred_col].max()
        row["cash"] = 0.065 / 252
        row["nifty"] = row.get("nifty_fwd_5", np.nan)
        row["defensive"] = defensive["ret_fwd_5"].mean()
        row["cyclical"] = cyclical["ret_fwd_5"].mean()
        row["low_risk"] = low["ret_fwd_5"].mean()
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .dropna(subset=["cash", "nifty", "defensive", "cyclical", "low_risk"])
        .sort_values("date")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_bandit(train_rewards, test_rewards, context_cols, cfg: Config):
    """
    Warm up the bandit on training data then evaluate on test data.

    Parameters
    ----------
    train_rewards : pd.DataFrame  – output of make_bandit_frame (train period)
    test_rewards  : pd.DataFrame  – output of make_bandit_frame (test period)
    context_cols  : list[str]     – columns used as context features
    cfg           : Config

    Returns
    -------
    pd.DataFrame with columns:
        date, action, reward, oracle_reward, regret, cost
    """
    actions = ["cash", "nifty", "defensive", "cyclical", "low_risk"]
    scaler = StandardScaler().fit(
        train_rewards[context_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    )

    def ctx(df):
        X = scaler.transform(
            df[context_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        )
        return np.c_[np.ones(len(X)), X]

    bandit = BayesianLinearBandit(
        actions, len(context_cols) + 1, random_state=cfg.random_state
    )

    # Warm-up on training period
    warm = train_rewards.iloc[:: cfg.rebalance_step].reset_index(drop=True)
    Xw = ctx(warm)
    for i, row in warm.iterrows():
        a = bandit.choose(Xw[i])
        bandit.update(a, Xw[i], row[a])

    # Evaluate on test period
    te = test_rewards.iloc[:: cfg.rebalance_step].reset_index(drop=True)
    Xt = ctx(te)
    rec = []
    last = None
    for i, row in te.iterrows():
        a = bandit.choose(Xt[i])
        cost = cfg.transaction_cost if last is not None and a != last else 0.0
        r = row[a] - cost
        bandit.update(a, Xt[i], r)
        oracle = max(row[x] for x in actions)
        rec.append({
            "date": row["date"],
            "action": a,
            "reward": r,
            "oracle_reward": oracle,
            "regret": oracle - r,
            "cost": cost,
        })
        last = a

    return pd.DataFrame(rec)
