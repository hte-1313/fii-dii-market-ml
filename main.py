# =============================================================================
# main.py
# FII-DII Market ML Pipeline — full end-to-end run
# =============================================================================
# Imports from:
#   models/config.py
#   models/data.py
#   models/features.py
#   models/ols_ridge_lasso.py
#   models/logistic.py
#   models/xgboost_model.py
#   models/bayesian_ridge.py
#   models/bayesian_logistic_laplace.py
#   models/market_regime_hmm.py
#   models/game_response.py
#   models/pymc_hierarchical.py
#   models/ensemble_benchmarks.py
#   models/bandit.py
# =============================================================================

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (11, 5)

# Add models/ to path so imports resolve
sys.path.insert(0, str(Path(__file__).parent / "models"))

from config import get_config
from data import DataManager
from features import FeatureBuilder, DesignMatrix, ModelMetrics
from ols_ridge_lasso import compare_regression_models, run_ols, run_ridge, run_lasso
from logistic import run_logistic
from xgboost_model import run_xgboost
from bayesian_ridge import run_bayesian_ridge
from bayesian_logistic_laplace import run_bayesian_logistic_laplace
from market_regime_hmm import MarketRegimeModel
from game_response import GameResponseLearner
from pymc_hierarchical import PyMCHierarchicalGameModel
from ensemble_benchmarks import run_all_benchmarks
from bandit import make_bandit_frame, run_bandit


# =============================================================================
# SECTION 0 — Configuration
# =============================================================================

cfg = get_config()


# =============================================================================
# SECTION 1 — Data loading and source validation
# =============================================================================
#
# Loads two datasets:
#   prices : daily market, sector-index, and volatility data
#   flows  : daily FII/FPI and DII institutional-flow data
#
# The DataManager tries NSE first and falls back to Yahoo Finance.
# Any known stale cached files are removed before loading.
#
# prices shape = (n_price_rows, n_price_columns)
# flows  shape = (n_flow_rows,  n_flow_columns)

bad_files = [
    Path(cfg.data_dir) / "bad_latest_fii_dii.csv",
    Path(cfg.data_dir) / "index_ENERGY.csv",
]
for f in bad_files:
    if f.exists():
        f.unlink()

data = DataManager(cfg)
prices, flows, source = data.load_all()
print("Data source:", source)

price_src = source.get("prices", {}).get("source", "unknown")
failures  = source.get("prices", {}).get("index_failures", [])
if "yahoo" in price_src:
    print(f"\nWARNING: NSE scraper failed for {len(failures)} index/indices.")
    print(f"  Failed: {[f[0] for f in failures]}")
    print(f"  All price data loaded from Yahoo Finance (fallback).\n")
else:
    print("\nPrice data loaded directly from NSE.\n")

print("Prices:", prices.shape)
print("Flows:", flows.shape)


# =============================================================================
# SECTION 2 — Feature construction and data audit
# =============================================================================
#
# The empirical structure is a sector-day panel:
#   (s, t),  s = 1..S sectors,  t = 1..T trading days
#
# Each row contains market features, sector features, institutional-flow
# variables, and a forward downside label.
#
# Classification target:
#   y_{s, t+h} = 1  if sector s experiences a downside event over horizon h
#   y_{s, t+h} = 0  otherwise
#
# Chronological split:
#   D_train = {(s,t) : t <= t_cut}
#   D_test  = {(s,t) : t >  t_cut}
#
# Random splitting would leak future information and overstate performance.

builder = FeatureBuilder(cfg)
daily, panel, sectors, price_table = builder.build(prices, flows)
cut = builder.split_date(panel)
panel = builder.label_downside(panel, cut)
train, test = builder.split(panel, cut)
daily_train, daily_test = builder.split(daily, cut)

print("Cut date:", cut)
print("Daily:", daily.shape)
print("Panel:", panel.shape)
print("Sectors:", sectors)


def audit_dataset(df, name, date_col="date"):
    out = {
        "dataset": name,
        "rows": len(df),
        "columns": df.shape[1],
        "start": pd.NaT,
        "end": pd.NaT,
        "unique_dates": np.nan,
        "missing_pct": df.isna().mean().mean(),
        "duplicate_rows": df.duplicated().sum(),
    }
    if date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce")
        out["start"] = d.min()
        out["end"]   = d.max()
        out["unique_dates"] = d.nunique()
    return out


audit_table = pd.DataFrame([
    audit_dataset(daily, "daily_market_flow"),
    audit_dataset(panel, "sector_panel"),
    audit_dataset(train, "train_panel"),
    audit_dataset(test,  "test_panel"),
])
print(audit_table)


# =============================================================================
# SECTION 3 — Exploratory data analysis
# =============================================================================
#
# Downside rate by sector:
#   y_bar_s = (1/T_s) * sum_{t=1}^{T_s} y_{s, t+h}
#
# This is the empirical frequency with which sector s experiences a future
# downside event. Differences across sectors motivate the hierarchical model.
#
# Cumulative NIFTY return:
#   R_t^{cum} = sum_{tau <= t} r_tau^{NIFTY}
#
# Standardised institutional flow pressure:
#   FII_z_t = (FII_t - mu_FII) / sigma_FII
#   DII_z_t = (DII_t - mu_DII) / sigma_DII
#
# Sharp negative FII + positive DII = domestic absorption of foreign selling.
# Joint negative = institutional stress signal.

print(daily.describe().T)
print(panel.groupby("sector")["downside"].mean().sort_values().to_frame("downside_rate"))

plt.figure(figsize=(12, 4))
plt.plot(daily["date"], daily["nifty_r1"].cumsum())
plt.title("NIFTY cumulative daily return")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(daily["date"], daily["fii_z"], label="FII z")
plt.plot(daily["date"], daily["dii_z"], label="DII z")
plt.title("Institutional flow pressure")
plt.legend()
plt.tight_layout()
plt.show()

heat_cols = [
    "nifty_r1", "nifty_r5", "rv_20", "drawdown",
    "sector_dispersion", "vix_chg", "fii_z", "dii_z", "flow_imbalance",
]
plt.figure(figsize=(9, 6))
sns.heatmap(daily[heat_cols].corr(), center=0, cmap="coolwarm", annot=False)
plt.title("Feature correlation heatmap")
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 4 — Feature sets and design matrices
# =============================================================================
#
# Baseline feature vector:
#   X^{base}_{s,t} = [r^{NIFTY}_t, r^{NIFTY}_{t-5:t}, RV_{20,t}, DD_t,
#                     Disp_t, delta_VIX_t, FII_z_t, DII_z_t,
#                     FlowImb_t, r_{s,t}, r_{s,t-5:t}]
#
# Full feature vector (augmented with game-theoretic variables):
#   X^{full}_{s,t} = [X^{base}_{s,t}, G_t]
#
# Classification target:
#   y_{s, t+h} in {0, 1}
#
# Train event rate:  p_hat_train = (1/n_train) * sum_i y_i
# Test  event rate:  p_hat_test  = (1/n_test)  * sum_i y_i
#
# Large differences between these rates indicate different risk environments
# across the two periods, which should be discussed in results interpretation.

base_cols = [
    "nifty_r1", "nifty_r5", "rv_20", "drawdown", "sector_dispersion",
    "vix_chg", "fii_z", "dii_z", "flow_imbalance", "sector_r1", "sector_r5",
]
game_cols = [
    "fii_action", "dii_action", "joint_buy", "joint_sell",
    "absorption", "contested", "action_alignment",
]
all_cols = base_cols + game_cols

dm_base = DesignMatrix(base_cols)
X_train_base = dm_base.fit_transform(train)
X_test_base  = dm_base.transform(test)

dm_all = DesignMatrix(all_cols)
X_train_all = dm_all.fit_transform(train)
X_test_all  = dm_all.transform(test)

y_train = train["downside"].values
y_test  = test["downside"].values
r_train = train["ret_fwd_5"].values
r_test  = test["ret_fwd_5"].values

print("Train shape:", train.shape, "  Test shape:", test.shape)
print("Train event rate:", y_train.mean())
print("Test  event rate:", y_test.mean())


# =============================================================================
# SECTION 5.1 — Baseline return regression: OLS
# =============================================================================
#
# Model:
#   r_{s, t+5} = alpha + X^{base}_{s,t}' beta + epsilon_{s,t}
#
# OLS estimator:
#   beta_hat_OLS = argmin_beta  sum_i (r_i - X_i' beta)^2
#
# Evaluation on the chronological test set using RMSE, MAE, R^2.
# This is the simplest linear benchmark for return prediction.

_, ols_pred, ols_metrics = run_ols(X_train_base, r_train, X_test_base, r_test)
print(pd.DataFrame([ols_metrics], index=["OLS"]))


# =============================================================================
# SECTION 5.2 — Regularised return baselines: Ridge and Lasso
# =============================================================================
#
# Ridge:
#   beta_hat_Ridge = argmin_beta [ sum_i (r_i - X_i'beta)^2
#                                  + lambda * sum_j beta_j^2 ]
#   L2 penalty shrinks all coefficients but keeps every predictor active.
#
# Lasso:
#   beta_hat_Lasso = argmin_beta [ sum_i (r_i - X_i'beta)^2
#                                  + lambda * sum_j |beta_j| ]
#   L1 penalty can set some coefficients exactly to zero (feature selection).
#
# If all three perform similarly, this motivates moving from exact return
# prediction to probabilistic downside-risk classification.

regression_table = compare_regression_models(X_train_base, r_train, X_test_base, r_test)
print(regression_table)


# =============================================================================
# SECTION 5.3 — Baseline downside classifier: Logistic regression
# =============================================================================
#
# Model:
#   logit(p_{s,t}) = log(p / (1-p)) = alpha + X^{base}_{s,t}' beta
#   p_{s,t} = 1 / (1 + exp(-(alpha + X^{base}_{s,t}' beta)))
#
# y_{s, t+5} ~ Bernoulli(p_{s,t})
#
# class_weight='balanced' prevents the classifier from predicting only the
# majority class when downside events are less frequent.
#
# Key metrics: log loss and Brier score (calibrated probability quality),
# AUC (discrimination), F1, precision, recall.

logit_model, logit_p, logit_metrics = run_logistic(
    X_train_base, y_train, X_test_base, y_test, threshold=y_train.mean()
)
print(pd.DataFrame([logit_metrics], index=["Logistic"]))


# =============================================================================
# SECTION 5.4 — Nonlinear downside classifier: XGBoost
# =============================================================================
#
# XGBoost builds an additive ensemble of decision trees:
#   p_hat_{s,t} = sigma( sum_{m=1}^{M} f_m(X^{full}_{s,t}) )
#
# where each f_m is a regression tree, sigma(.) is the logistic function.
#
# Objective:
#   L = sum_i l(y_i, p_hat_i) + sum_{m=1}^{M} Omega(f_m)
#
# l(.) = log-loss classification objective
# Omega(f_m) = tree complexity penalty (regularisation)
#
# scale_pos_weight adjusts for class imbalance.
# Fitted on the full feature set X^{full} = [X^{base}, G_t].

xgb_model, xgb_p, xgb_metrics = run_xgboost(
    X_train_all, y_train, X_test_all, y_test,
    random_state=cfg.random_state, threshold=y_train.mean(),
)
print(pd.DataFrame([xgb_metrics], index=["XGBoost"]))


# =============================================================================
# SECTION 5.5 — Bayesian Ridge return regression
# =============================================================================
#
# Bayesian Ridge places a prior on regression coefficients and returns
# predictive uncertainty (standard deviation) alongside point estimates.
# Predictive intervals are plotted for a visual calibration check.

_, bayes_ridge_pred, bayes_ridge_std, bayes_ridge_metrics = run_bayesian_ridge(
    X_train_base, r_train, X_test_base, r_test, plot=True
)
print(pd.DataFrame([bayes_ridge_metrics], index=["Bayesian Ridge"]))


# =============================================================================
# SECTION 5.6 — Bayesian downside classifier: Laplace logistic regression
# =============================================================================
#
# Model:
#   y_{s, t+5} ~ Bernoulli(p_{s,t})
#   logit(p_{s,t}) = alpha + X^{base}_{s,t}' beta
#   beta ~ N(0, tau^2 * I)
#
# Posterior approximated as N(beta_hat, H^{-1}) via the Laplace method,
# where H is the Hessian of the negative log-posterior at the MAP estimate.
#
# Predicted probabilities averaged over posterior coefficient draws:
#   p_hat_{s,t} = (1/M) * sum_{m=1}^{M} sigma(alpha^(m) + X' beta^(m))

bayes_logit, bayes_p, bayes_metrics = run_bayesian_logistic_laplace(
    X_train_base, y_train, X_test_base, y_test,
    prior_var=1.0, draws=1000, random_state=cfg.random_state,
    threshold=y_train.mean(),
)
print(pd.DataFrame([bayes_metrics], index=["Bayesian Logistic Laplace"]))


# =============================================================================
# SECTION 5.7 — Hidden-state market regime layer (Gaussian HMM)
# =============================================================================
#
# The market switches between K hidden states z_t in {1, 2, 3}.
#
# State transition (Markov chain):
#   P(z_t = j | z_{t-1} = i) = Pi_{ij}
#
# Observation model (Gaussian emissions):
#   X_t | z_t = k  ~  N(mu_k, Sigma_k)
#
# Filtered regime probabilities (causal — only past information used):
#   P(z_t = k | X_{1:t})
#
# States are labelled risk_on / contested / stress by sorting on a stress
# score derived from returns, VIX changes, and realised volatility.
#
# Regime entropy:
#   H_t = -sum_{k=1}^{K} p_{t,k} * log(p_{t,k})
# Lower entropy = more confident regime assignment.

regime_cols = [
    "nifty_r1", "vix_chg", "rv_20", "drawdown",
    "fii_z", "dii_z", "flow_imbalance",
]
regime_model = MarketRegimeModel(n_regimes=3, random_state=cfg.random_state)
regime_model.fit(daily_train, regime_cols)
regimes = regime_model.filtered_probabilities(daily)

daily_regime = daily.merge(regimes, on="date", how="left")
panel_regime  = panel.merge(regimes, on="date", how="left")
regime_prob_cols = [
    c for c in panel_regime.columns
    if c.startswith("regime_") and c not in ("regime_label", "regime_entropy")
]

print(daily_regime.groupby("regime_label")[regime_cols].mean())
regime_model.plot_regimes(daily_regime)
regime_model.diagnostics(daily_regime)


# =============================================================================
# SECTION 5.8 — Game-theoretic institutional response layer
# =============================================================================
#
# Each institution is assigned a discrete action:
#   a^{FII}_t, a^{DII}_t  in {-1, 0, 1}
#   -1 = net selling,  0 = neutral,  1 = net buying
#
# The model estimates the empirical response distribution:
#   P(a^{DII}_t = j | a^{FII}_t = i)  for i, j in {-1, 0, 1}
#
# Derived strategic-risk features:
#   Domestic Absorption_t = I(FII sells) * P(DII buys)
#   Joint Stress_t        = I(FII sells) * P(DII sells)
#
# These features encode repeated strategic interaction patterns between
# foreign and domestic institutional investors.

daily_train_r, daily_test_r = builder.split(daily_regime, cut)
response_learner = GameResponseLearner().fit(daily_train_r)
daily_game = response_learner.transform(daily_regime)
panel_game = panel_regime.merge(
    daily_game[[
        "date", "p_dii_sell_resp", "p_dii_neutral_resp",
        "p_dii_buy_resp", "p_domestic_absorption", "p_joint_stress",
    ]],
    on="date", how="left",
)

response_learner.plot_response_heatmap(daily_game)


# =============================================================================
# SECTION 5.9a — Bayesian game-regime downside classifier (Laplace)
# =============================================================================
#
# Full feature vector:
#   X^{game-regime}_{s,t} = [X^{base}_{s,t}, G_t, Z_t, R_t]
#
# G_t = direct FII/DII action variables
# Z_t = filtered regime probabilities P(z_t = stress | X_{1:t})
# R_t = learned DII response probabilities P(a^{DII} = j | a^{FII}, z_t)
#
# Model:
#   y_{s, t+5} ~ Bernoulli(p_{s,t})
#   logit(p_{s,t}) = alpha + X^{game-regime}_{s,t}' beta
#   beta ~ N(0, tau^2 * I)
#
# Posterior approximated via Laplace method.
# Tests whether augmenting market variables with regime + institutional
# interaction improves calibrated downside-risk forecasting.

train_r, test_r = builder.split(panel_game.dropna(), cut)
response_cols = ["p_dii_sell_resp", "p_dii_buy_resp"]
full_cols = all_cols + regime_prob_cols + response_cols

dm_full = DesignMatrix(full_cols)
X_train_full = dm_full.fit_transform(train_r)
X_test_full  = dm_full.transform(test_r)
y_train_r = train_r["downside"].values
y_test_r  = test_r["downside"].values

bayes_game, bayes_game_p, bayes_game_metrics = run_bayesian_logistic_laplace(
    X_train_full, y_train_r, X_test_full, y_test_r,
    prior_var=1.0, draws=1000, random_state=cfg.random_state,
    threshold=y_train_r.mean(),
)
bayes_game_train_p = bayes_game.predict_proba(X_train_full)[:, 1]
print(pd.DataFrame([bayes_game_metrics], index=["Bayesian Game-Regime Laplace"]))


# =============================================================================
# SECTION 5.9b — Advanced Bayesian model: PyMC hierarchical game-regime
# =============================================================================
#
# Full hierarchical Bayesian logistic regression:
#   y_{s, t+5} ~ Bernoulli(p_{s,t})
#   logit(p_{s,t}) = alpha_s + X_{s,t}' beta
#
# Sector intercepts (partial pooling):
#   alpha_s  ~ N(mu_alpha, sigma_alpha^2)
#   mu_alpha ~ N(0, 1)
#   sigma_alpha ~ HalfNormal(1)
#
# Coefficient shrinkage (local-global):
#   beta_j   ~ N(0, tau^2 * lambda_j^2)
#   tau      ~ HalfCauchy(1)
#   lambda_j ~ HalfCauchy(1)
#
# Posterior inference via PyMC NUTS.
# Posterior predictive mean:
#   p_hat_{s,t} = (1/M) * sum_{m=1}^{M} sigma(alpha_s^(m) + X' beta^(m))
#
# This is the most complete Bayesian specification: sector partial pooling +
# institutional game variables + hidden-regime probabilities + posterior
# uncertainty quantification.

n_full = len(train_r)
n_cap  = cfg.pymc_max_rows
print(
    f"PyMC model will be fitted on {min(n_cap, n_full):,} of {n_full:,} "
    f"training rows ({100 * min(n_cap, n_full) / n_full:.1f}%) "
    f"due to pymc_max_rows = {n_cap}."
)

pymc_model = PyMCHierarchicalGameModel(cfg)
pymc_model.fit(train_r, full_cols)

pymc_train_p = pymc_model.predict_proba(train_r)
pymc_p       = pymc_model.predict_proba(test_r)

valid = ~np.isnan(pymc_p)
pymc_metrics = ModelMetrics.classification(
    y_test_r[valid], pymc_p[valid], threshold=y_train_r.mean()
)
print(pd.DataFrame([pymc_metrics], index=["PyMC Hierarchical Game-Regime Shrinkage"]))
print(az.summary(pymc_model.idata, var_names=["mu_alpha", "sigma_alpha", "tau"], round_to=3))
print(pymc_model.coefficient_summary().head(15))
print(pymc_model.sampling_diagnostics())

# Coefficient plot
coef_table = pymc_model.coefficient_summary()
top_coef = coef_table.head(12).sort_values("mean")
plt.figure(figsize=(9, 5))
plt.errorbar(
    top_coef["mean"], top_coef["feature"],
    xerr=[top_coef["mean"] - top_coef["hdi_3%"], top_coef["hdi_97%"] - top_coef["mean"]],
    fmt="o",
)
plt.axvline(0, linewidth=1)
plt.title("PyMC posterior coefficients with 94% HDI")
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 6 — Additional benchmark models and fair model comparison
# =============================================================================
#
# Full feature set:
#   X^{full}_{s,t} = [X^{base}_{s,t}, G_t, Z_t, R_t]
#
# Tree ensemble models approximate:
#   p_hat_{s,t} = sigma( sum_{m=1}^{M} f_m(X^{full}_{s,t}) )  [boosting]
#   p_hat_{s,t} = (1/B) * sum_{b=1}^{B} p_hat^{(b)}_{s,t}      [forests]
#
# Tree models are calibrated with isotonic regression to correct
# miscalibrated probability outputs.
#
# Gaussian Process classifier (only for small training sets):
#   f(.) ~ GP(0, K),  K = RBF + DotProduct
#
# The comparison is split into:
#   (1) Base-feature models:  conventional market, sector, flow predictors only
#   (2) Full-feature models:  base + game variables + regime probabilities
#
# Fairness: base logistic should not compete against full-feature models.
# The key question within full-feature models:
#   Do Bayesian models, boosted trees, or GPs produce the best downside
#   risk probabilities given the same information?

spw = max(1, sum(y_train_r == 0) / max(1, sum(y_train_r == 1)))
xgb_full_model, xgb_p_full, xgb_metrics_full = run_xgboost(
    X_train_full, y_train_r, X_test_full, y_test_r,
    random_state=cfg.random_state, threshold=y_train_r.mean(),
)

_, ensemble_table = run_all_benchmarks(
    X_train_full, y_train_r, X_test_full, y_test_r,
    random_state=cfg.random_state, threshold=y_train_r.mean(),
)

# Full comparison table
forecast_table = pd.DataFrame(
    [logit_metrics, bayes_metrics, bayes_game_metrics, pymc_metrics, xgb_metrics_full],
    index=[
        "Logistic", "Bayesian Logistic Laplace",
        "Bayesian Game-Regime Laplace", "PyMC Hierarchical Game-Regime",
        "XGBoost (full features)",
    ],
)
forecast_table = pd.concat([forecast_table, ensemble_table])
print("--- All models (sorted by LogLoss) ---")
print(forecast_table.sort_values("LogLoss"))

# Calibration check
for label, probs in [("Bayesian Game-Regime", bayes_game_p), ("PyMC", pymc_p)]:
    tmp = pd.DataFrame({"y": y_test_r, "p": probs}).dropna()
    tmp["bin"] = pd.qcut(tmp["p"], 10, duplicates="drop")
    cal_table = tmp.groupby("bin").agg(
        mean_pred=("p", "mean"), realised=("y", "mean"), n=("y", "size")
    ).reset_index()
    plt.figure(figsize=(5, 5))
    plt.plot(cal_table["mean_pred"], cal_table["realised"], marker="o", label=label)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"Calibration: {label}")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Realised downside rate")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 7 — Bayesian contextual bandit decision layer
# =============================================================================
#
# At each rebalance date t, the bandit observes context vector c_t:
#   c_t = [mean_downside_prob, max_downside_prob, delta_VIX_t, RV_{20,t},
#           DD_t, FII_z_t, DII_z_t, flow_imbalance_t,
#           P(z_t = k), absorption_t, joint_stress_t]
#
# Available actions:
#   a_t in {cash, NIFTY, defensive, cyclical, low_risk}
#
# Linear reward model per action:
#   E[r_{t+1}(a) | c_t] = c_t' theta_a
#
# Posterior over theta_a updated after each observed reward (Thompson sampling).
#
# Realised reward:
#   r_t(a_t) = R_t(a_t) - transaction_cost_t
#
# Cumulative regret:
#   Regret_T = sum_{t=1}^{T} [ r_t(a_t*) - r_t(a_t) ]
# where a_t* = best action in hindsight at time t.
#
# Sharpe ratio:
#   Sharpe = E[R_p] / sigma(R_p)
#
# Max Drawdown:
#   MaxDD = min_t ( W_t / max_{tau<=t} W_tau  - 1 )
#
# The oracle benchmark (best action in hindsight) is infeasible in practice
# and is included only as a diagnostic upper bound.

train_r = train_r.copy()
test_r  = test_r.copy()
train_r["p_model"] = (
    pymc_train_p if len(pymc_train_p) == len(train_r) else bayes_game_train_p
)
test_r["p_model"] = np.where(np.isnan(pymc_p), bayes_game_p, pymc_p)

train_rewards = make_bandit_frame(train_r, daily_game, "p_model", cfg)
test_rewards  = make_bandit_frame(test_r,  daily_game, "p_model", cfg)

context_cols = [
    "mean_downside_prob", "max_downside_prob", "vix_chg", "rv_20", "drawdown",
    "fii_z", "dii_z", "flow_imbalance",
    "regime_risk_on", "regime_contested", "regime_stress",
    "p_domestic_absorption", "p_joint_stress",
]
context_cols = [c for c in context_cols if c in train_rewards.columns]
bandit_result = run_bandit(train_rewards, test_rewards, context_cols, cfg)

actions = ["cash", "nifty", "defensive", "cyclical", "low_risk"]
bench = pd.DataFrame({
    "Bandit": bandit_result["reward"],
    "OracleDiagnostic": bandit_result["oracle_reward"],
})
for a in actions:
    bench[a] = test_rewards.iloc[:: cfg.rebalance_step].reset_index(drop=True)[a]

strategy_table = pd.DataFrame({
    c: ModelMetrics.strategy(bench[c], periods=252 // cfg.rebalance_step)
    for c in bench.columns
}).T
print(strategy_table)
print(bandit_result["action"].value_counts().to_frame("n"))

# Wealth and regret plots
wealth = (1 + bench).cumprod()
plt.figure(figsize=(12, 5))
for c in wealth.columns:
    plt.plot(bandit_result["date"], wealth[c], label=c)
plt.title("Bayesian contextual bandit wealth curve")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(bandit_result["date"], bandit_result["regret"].cumsum())
plt.title("Cumulative regret diagnostic")
plt.tight_layout()
plt.show()


# =============================================================================
# SECTION 8 — Final model dashboard
# =============================================================================
#
# Return prediction (RMSE):
#   RMSE = sqrt( (1/n) * sum_i (r_i - r_hat_i)^2 )
#   Compares OLS vs Bayesian Ridge.
#
# Downside-risk classification (LogLoss):
#   LogLoss = -(1/n) * sum_i [ y_i*log(p_hat_i) + (1-y_i)*log(1-p_hat_i) ]
#   Lower = better calibrated probability forecasts.
#
# Strategy performance:
#   Sharpe  = E[R_p] / sigma(R_p)
#   MaxDD   = min_t ( W_t / max_{tau<=t} W_tau - 1 )
#
# The dashboard layers results:
#   Return models    → is next-week sector return linearly predictable?
#   Classification   → can downside risk be forecast probabilistically?
#   PyMC model       → does hierarchy + regimes + game variables improve it?
#   Bandit results   → can the forecasts support a sequential allocation rule?

dashboard = pd.DataFrame({
    "OLS_RMSE":              [ols_metrics["RMSE"]],
    "Bayesian_Ridge_RMSE":   [bayes_ridge_metrics["RMSE"]],
    "Logistic_LogLoss":      [logit_metrics["LogLoss"]],
    "Bayesian_LogLoss":      [bayes_metrics["LogLoss"]],
    "Game_Regime_LogLoss":   [bayes_game_metrics["LogLoss"]],
    "PyMC_LogLoss":          [pymc_metrics["LogLoss"]],
    "Bandit_Sharpe":         [strategy_table.loc["Bandit", "Sharpe"]],
    "Nifty_Sharpe":          [
        strategy_table.loc["nifty", "Sharpe"]
        if "nifty" in strategy_table.index else np.nan
    ],
    "Bandit_MaxDD":          [strategy_table.loc["Bandit", "MaxDD"]],
}).T.rename(columns={0: "value"})

print("\n========== FINAL DASHBOARD ==========")
print(dashboard)
