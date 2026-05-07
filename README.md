# Bayesian Game-Regime Downside Risk Model 

A probabilistic forecasting pipeline for sector-level downside risk in Indian equity markets. The project layers three ideas on top of each other: hidden Markov market regimes, a game-theoretic model of institutional investor interactions, and a hierarchical Bayesian classifier that fuses both into calibrated downside-risk probabilities. A contextual bandit then converts those probabilities into actual allocation decisions.

The central question is: *do foreign (FII) and domestic (DII) institutional flows, when modelled as a strategic interaction rather than two independent predictors, improve calibrated sector downside-risk forecasts on the NSE?*

---

## What the Project Does

The pipeline covers nine NSE sector indices (BANK, IT, FMCG, AUTO, PHARMA, METAL, REALTY, FIN_SERVICE, and the broad NIFTY 50) plus India VIX, using daily data from 2015 onwards. It is structured as a sequence of modelling layers that progressively enrich a base feature set.

### The Full Modelling Stack

| Layer | Model | Purpose |
|-------|-------|---------|
| **Baseline regression** | OLS, Ridge, Lasso | 5-day forward return point prediction — benchmark only |
| **Baseline classification** | Logistic Regression, XGBoost | Downside probability without regime or flow structure |
| **Bayesian regression** | Bayesian Ridge | Posterior uncertainty on return forecasts |
| **Bayesian classification** | Laplace logistic | Calibrated downside probability with uncertainty |
| **Regime layer** | Gaussian HMM (3 states) | Filtered probabilities of *risk-on*, *contested*, and *stress* regimes |
| **Game layer** | Empirical response learner | Conditional DII response distributions given FII action |
| **Main model** | Hierarchical Bayesian logistic regression | Sector-specific intercepts + full game-regime feature set, estimated via MCMC (PyMC) |
| **Decision layer** | Bayesian contextual bandit | Sequential allocation rule conditioned on regime and flow context |
| **Benchmarks** | Random Forest, Extra Trees, HistGradBoost, GPC | Fair nonlinear comparison against the Bayesian pipeline |

---

## What Makes This Different

### The Game-Theoretic Layer

The typical approach treats FII and DII net flows as two separate numerical predictors. This project treats them as a repeated institutional interaction. Each institution is assigned a discrete action — sell (−1), neutral (0), or buy (+1) — and the model estimates the empirical conditional response distribution:

$$P(a^{DII}_t = j \mid a^{FII}_t = i)$$

This produces four interpretable interaction regimes:
- **Joint buying** (FII=+1, DII=+1): coordinated demand pressure
- **Absorption** (FII=−1, DII=+1): domestic institutions absorbing foreign selling
- **Contested** (FII=+1, DII=−1): opposing institutional positioning
- **Joint stress** (FII=−1, DII=−1): coordinated institutional selling

The learned response probabilities — e.g., *P(DII sells | FII sells, regime = stress)* — become features in the main classifier, so the model is explicitly told about the strategic interaction, not just the raw flows.

### The Hidden Regime Layer

A Gaussian HMM is fitted on daily market variables (returns, VIX change, realised vol, drawdown, flow z-scores, flow imbalance). The three latent states are interpreted as market regimes, and the filtered state probabilities are passed downstream as features. This means the main classifier sees *P(stress regime today)* as an input, not just the raw market conditions.

### The Hierarchical Bayesian Model

The main model is a hierarchical Bayesian logistic regression estimated via MCMC using PyMC. Sector-specific intercepts are partially pooled:

$$\alpha_s \sim N(\mu_\alpha, \sigma_\alpha^2)$$

so sectors with less data borrow strength from the overall market-level estimate rather than being fit independently. The full feature vector fuses base market features, FII/DII game variables, HMM regime probabilities, and learned DII response probabilities. The output is a full posterior distribution over downside probability, not a point estimate.

---

## Data Sources

- **NSE India** index history via the NSE API (chunked 360-day requests), with automatic `yfinance` fallback for each sector index
- **India VIX** from NSE historical VIX data (or `^INDIAVIX` via yfinance)
- **FII/DII daily flows** from the NSE `fiidiiTradeReact` endpoint — if unavailable, the pipeline generates synthetic flows from a return-stress model for development purposes (flagged clearly in the data audit)

The pipeline validates that FII/DII data covers at least 250 unique trading dates before proceeding; if not, it either raises an error or falls back to demo mode depending on the config flag.

---

## Evaluation

Models are compared on a chronological train/test split (75%/25% by date, no shuffling). Key metrics:

- **Log loss** and **Brier score** — calibration quality of the downside probability estimates
- **ROC-AUC** and **average precision** — ranking ability
- **RMSE** — for the return regression baselines
- **Bandit cumulative reward** — for the allocation layer

Tree models (RF, Extra Trees, HistGradBoost) are calibrated with isotonic regression before comparison, so the evaluation is fair on probabilistic grounds.

---

## Limitations

**FII/DII data availability.** The NSE public API is unofficial and has no guaranteed uptime or schema stability. The pipeline handles this gracefully with a yfinance fallback and a demo-flow generator, but results in production depend entirely on whether the NSE endpoint is accessible and returns a parseable response.

**The game-theoretic layer is empirical, not structural.** The FII-DII interaction is modelled as an observed response distribution estimated from data, not as a formal game-theoretic equilibrium. There is no optimisation over strategies or utility functions — it is a structured featurisation of the joint flow pattern rather than a true game-theoretic model in the economic sense.

**Three HMM states is an assumption.** The number of latent regimes is fixed at three for interpretability. The model doesn't select the number of states or validate that three is optimal — it just names them post-hoc based on the means of the fitted Gaussians.

**MCMC at scale is slow.** The PyMC hierarchical model caps the panel at 6,000 rows by default to keep inference tractable. For larger datasets, approximate inference (ADVI or Laplace approximation) would be needed, but hasn't been implemented here.

**The bandit reward signal is simplified.** The contextual bandit uses a stylised reward — sector return minus transaction cost — rather than a properly risk-adjusted performance measure. There is no portfolio-level constraint, no volatility targeting, and position sizes are discrete actions rather than continuous weights.

**No walk-forward refitting.** The entire pipeline uses a single train/test split. The HMM, game response learner, and Bayesian model are all fitted once on training data and applied statically to the test set. In practice, these would need periodic refitting as regime structure and institutional behaviour shift over time.

---

## Future Work

This is where things get interesting. The current pipeline is a proof of concept — it shows that game-regime features improve calibrated downside probability estimates over baseline classifiers. But there are several directions that would make this meaningfully more rigorous and more useful.

### 1. Walk-Forward Bayesian Inference with Sequential Updating

The biggest gap right now is that the model is static. A proper implementation would refit everything on a rolling or expanding window — ideally with sequential Bayesian updating rather than starting from scratch each window.

For the Bayesian logistic model, this means using the previous posterior as the prior for the next window. PyMC supports this in principle, but it requires serialising and reloading the posterior in a form that works as a prior specification for the next fit. The payoff is real: posterior uncertainty would actually reflect how confident the model is after seeing recent data, rather than treating 2015–2024 as a homogeneous block.

For the HMM, the natural extension is online regime tracking with the forward algorithm applied to new data, rather than rerunning `fit` from scratch each time. This would make the filtered state probabilities genuinely real-time rather than retrospective. The full vision here is a daily inference pipeline: new flow data arrives, the HMM updates the current regime probability, the game layer refreshes the conditional response features, and the Bayesian posterior is updated before the bandit makes its allocation decision.

### 2. A Proper Structural Game-Theoretic Model

The current game layer is really just a structured featurisation — it learns empirical response frequencies but doesn't optimise any strategy or derive any equilibrium. The natural next step is to model FII and DII as players in a repeated game with an explicit payoff structure.

One direction is to test whether observed FII-DII action pairs are consistent with a correlated equilibrium given the market environment. This would mean specifying a reward function for each player (e.g., FII maximises short-run alpha, DII maximises portfolio stabilisation) and checking whether observed joint actions satisfy the deviation-proof conditions. Another direction is inverse reinforcement learning: infer what reward functions rationalise the observed behaviour, which would give a much richer model of *why* institutions behave the way they do under different regimes, not just *what* they tend to do.

Either approach would make the game features principled rather than empirical, and would give much more interpretable coefficients in the downstream Bayesian model.

### 3. Dynamic and Non-Parametric Regime Detection

Three fixed HMM states with Gaussian emissions is a deliberately simple model of market dynamics. There are several well-studied extensions worth trying:

A **non-parametric number of states** via the Dirichlet process HMM (infinite HMM) would let the data determine how many regimes are needed rather than imposing three up front. Practically, this would be implemented via a stick-breaking prior on the transition distribution — the model can create new states when the existing ones don't describe the data well.

**Non-Gaussian emissions** would handle fat-tailed return days without mislabelling them as regime switches. A Student-t emission model, with learnable degrees-of-freedom per state, would make the regime detection much more robust to the kind of extreme moves that appear around RBI announcements and global risk events.

**Input-dependent transition probabilities** — where the probability of switching regime is a function of VIX level or flow imbalance rather than a fixed matrix — would capture the well-known empirical pattern that high-volatility periods are themselves persistent. This is the Markov-switching regression extension and would fit naturally into the existing framework.

### 4. Full Posterior Predictive Portfolio Optimisation

Right now the bandit takes the *point estimate* of downside probability as its context and selects a discrete action. A richer approach would propagate the full posterior over downside probabilities into the allocation decision — treating portfolio construction as a Bayesian decision problem with an explicitly specified loss function.

In practice, this means drawing samples from the PyMC posterior predictive distribution and using those samples directly as inputs to a risk-aware optimiser (CVaR minimisation, for example), rather than collapsing the posterior to a mean and plugging it into a bandit. The bandit arm-selection step would then become Thompson sampling over a continuous weight space — much more natural for a proper portfolio problem.

This extension would also resolve the current limitation that position sizes are discrete actions: continuous weights, constrained to sum to one, with a penalty on portfolio volatility derived directly from the posterior predictive variance.

### 5. Sector Correlation Graph

The current model treats sectors as conditionally independent given the shared market features. But sectors are structurally connected — financials and real estate move together, metals and energy are correlated through commodity cycles, defensive sectors (FMCG, pharma) tend to decouple from cyclicals in stress. 

A graph-based extension would build a rolling sector correlation graph and propagate information across connected sectors using message passing (similar to the GCN layer in the ST446 pipeline). This would be especially useful for the game layer: if FII is heavily selling BANK, that signal should propagate to FIN_SERVICE and REALTY even if their own raw flow on that day looks neutral. The graph would also improve the bandit: allocating away from one sector while holding a correlated one defeats the purpose of the risk signal.

### 6. Options and Derivatives Data

India VIX is in the pipeline as a single aggregate implied volatility reading. The NSE options market has a lot more to offer — sector-level put-call ratios, the shape of the near-term volatility surface, and open interest patterns that often anticipate institutional flow reporting by a day or two.

Adding sector-level options data would require an additional data pipeline (the NSE API does expose derivative data), but the payoff for regime detection is meaningful: the options market tends to price stress events before they show up in spot-market flows, so leading VIX signals from the options surface could sharpen the timing of regime transitions in the HMM.

### 7. Posterior Interpretability and Sensitivity Analysis

The hierarchical Bayesian model produces full posterior distributions over every coefficient — which is genuinely valuable — but right now those posteriors are compressed into a summary dashboard and not really interrogated. 

Planned work: a proper posterior sensitivity analysis that asks which coefficients are stable across regimes, which ones flip sign in stress periods, and which sectors carry the widest posterior uncertainty. The game-theoretic features are particularly interesting here: does the absorption coefficient (FII selling absorbed by DII buying) have a consistently positive or negative effect on downside risk, or does that depend on the regime?

The joint flow pattern over time would also benefit from a dedicated visualisation — a rolling heatmap of the four institutional interaction states, overlaid with actual sector drawdowns, would make the game layer's contribution legible without needing to read off model coefficients.

### 8. Real-Time Inference Service

The pipeline currently lives in a notebook. The natural end state is a lightweight daily inference service: the NSE data ingestion (already chunked and fault-tolerant) runs on a schedule, the HMM and game response learner update, the PyMC model runs a short ADVI pass to refresh the posterior, and the bandit allocation action is exposed as an API response.

The main work is parameterising the refit cadence (probably daily for the bandit context, weekly for the full Bayesian model) and deciding how to handle the cold-start problem — what the model does on the first few days of a new regime before the posterior has updated significantly. FastAPI would be the natural serving layer; the PyMC model can be serialised to `arviz` InferenceData format and loaded without refitting on every request.

---

## Course Context

This project was submitted for ST451 Bayesian Machine Learning at the London School of Economics. The brief required demonstrated use of Bayesian inference — priors, posteriors, predictive distributions — and a meaningful comparison against non-Bayesian benchmarks. The Indian equity market application is genuine and the methodology is reasonably principled, but this is academic work. It is not investment advice and has not been tested in a live trading environment.
