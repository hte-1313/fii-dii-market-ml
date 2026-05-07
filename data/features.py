"""
features.py
-----------
FeatureBuilder  – builds the daily market table and sector-level panel.
DesignMatrix    – standardises and transforms features into model-ready arrays.
ModelMetrics    – regression, classification and strategy evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from config import Config


class FeatureBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def find_col(self, df, pats):
        low = {c: str(c).lower().replace(" ", "_") for c in df.columns}
        for p in pats:
            for c, l in low.items():
                if p.lower() in l:
                    return c
        return None

    def add_flow_features(self, df):
        out = df.copy()
        for c in ["fii_net", "dii_net"]:
            mu = out[c].rolling(252, min_periods=60).mean().shift(1)
            sd = out[c].rolling(252, min_periods=60).std().shift(1)
            out[c.replace("_net", "_z")] = (out[c] - mu) / (sd + 1e-8)
        out["fii_action"] = np.select(
            [out["fii_z"] > self.cfg.action_z, out["fii_z"] < -self.cfg.action_z],
            [1, -1],
            default=0,
        )
        out["dii_action"] = np.select(
            [out["dii_z"] > self.cfg.action_z, out["dii_z"] < -self.cfg.action_z],
            [1, -1],
            default=0,
        )
        out["flow_imbalance"] = (out["fii_net"] - out["dii_net"]) / (
            out["fii_net"].abs() + out["dii_net"].abs() + 1
        )
        out["flow_pressure"] = (
            (out["fii_net"].abs() + out["dii_net"].abs())
            .rolling(20, min_periods=5)
            .mean()
        )
        return out

    def add_game_features(self, df):
        out = df.copy()
        out["joint_buy"] = (
            (out["fii_action"] == 1) & (out["dii_action"] == 1)
        ).astype(int)
        out["joint_sell"] = (
            (out["fii_action"] == -1) & (out["dii_action"] == -1)
        ).astype(int)
        out["absorption"] = (
            (out["fii_action"] == -1) & (out["dii_action"] == 1)
        ).astype(int)
        out["contested"] = (
            (out["fii_action"] == 1) & (out["dii_action"] == -1)
        ).astype(int)
        out["action_alignment"] = out["fii_action"] * out["dii_action"]
        return out

    def build(self, prices, flows):
        prices = prices.copy()
        prices["date"] = pd.to_datetime(prices["date"])
        flows = flows.copy()
        flows["date"] = pd.to_datetime(flows["date"])

        prices = prices.sort_values("date").drop_duplicates("date").ffill()
        cols = [c for c in prices.columns if c not in ["date", "vix_close"]]

        if "NIFTY50" not in cols:
            raise ValueError("NIFTY50 column is required.")
        sectors = [c for c in cols if c != "NIFTY50"]
        if len(sectors) < 3:
            raise ValueError("At least three sector columns are required.")

        px = prices.set_index("date")[cols]
        r1 = px.pct_change()
        r5 = px.pct_change(5)
        fwd = px.pct_change(self.cfg.horizon).shift(-self.cfg.horizon)

        daily = pd.DataFrame(index=px.index)
        daily["nifty_close"] = px["NIFTY50"]
        daily["nifty_r1"] = r1["NIFTY50"]
        daily["nifty_r5"] = r5["NIFTY50"]
        daily["nifty_fwd_5"] = fwd["NIFTY50"]
        daily["rv_20"] = r1["NIFTY50"].rolling(20, min_periods=10).std() * np.sqrt(252)
        daily["drawdown"] = px["NIFTY50"] / px["NIFTY50"].expanding().max() - 1
        daily["sector_dispersion"] = r1[sectors].std(axis=1)

        daily = (
            daily.reset_index()
            .merge(prices[["date", "vix_close"]], on="date", how="left")
            .merge(flows, on="date", how="left")
            .sort_values("date")
        )
        daily["vix_close"] = daily["vix_close"].ffill()
        daily["vix_chg"] = daily["vix_close"].pct_change()
        daily[["fii_net", "dii_net"]] = daily[["fii_net", "dii_net"]].ffill()
        daily = self.add_game_features(self.add_flow_features(daily))

        frames = []
        for s in sectors:
            part = daily.copy()
            part["sector"] = s
            part["sector_r1"] = r1[s].reindex(pd.to_datetime(part["date"])).values
            part["sector_r5"] = r5[s].reindex(pd.to_datetime(part["date"])).values
            part["ret_fwd_5"] = fwd[s].reindex(pd.to_datetime(part["date"])).values
            frames.append(part)

        panel = pd.concat(frames, ignore_index=True)
        core = [
            "ret_fwd_5", "nifty_r1", "nifty_r5", "rv_20", "drawdown",
            "sector_dispersion", "vix_chg", "fii_z", "dii_z", "sector_r1", "sector_r5",
        ]
        daily = daily.dropna(
            subset=["nifty_r1", "nifty_r5", "rv_20", "drawdown", "sector_dispersion",
                    "vix_chg", "fii_z", "dii_z", "nifty_fwd_5"]
        ).reset_index(drop=True)
        panel = (
            panel.dropna(subset=core)
            .sort_values(["date", "sector"])
            .reset_index(drop=True)
        )
        return daily, panel, sectors, px.reset_index()

    def split_date(self, df):
        dates = sorted(pd.to_datetime(df["date"]).unique())
        cut = dates[int(len(dates) * (1 - self.cfg.test_size))]
        return cut

    def label_downside(self, panel, cut):
        out = panel.copy()
        train = out[out["date"] <= cut]
        q = (
            train.groupby("sector")["ret_fwd_5"]
            .quantile(self.cfg.downside_quantile)
            .to_dict()
        )
        out["downside_threshold"] = out["sector"].map(q)
        out["downside"] = (out["ret_fwd_5"] <= out["downside_threshold"]).astype(int)
        return out

    def split(self, df, cut):
        return df[df["date"] <= cut].copy(), df[df["date"] > cut].copy()


class DesignMatrix:
    def __init__(self, feature_cols):
        self.feature_cols = list(feature_cols)
        self.scaler = StandardScaler()

    def fit(self, df):
        self.sectors = sorted(df["sector"].dropna().unique()) if "sector" in df else []
        self.scaler.fit(
            df[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        )
        return self

    def transform(self, df, intercept=False):
        X = self.scaler.transform(
            df[self.feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        )
        if "sector" in df and self.sectors:
            S = (
                pd.get_dummies(df["sector"])
                .reindex(columns=self.sectors, fill_value=0)
                .values
            )
            X = np.c_[X, S]
        if intercept:
            X = np.c_[np.ones(len(X)), X]
        return X

    def fit_transform(self, df, intercept=False):
        self.fit(df)
        return self.transform(df, intercept=intercept)

    @property
    def names(self):
        return self.feature_cols + [f"sector_{s}" for s in self.sectors]


class ModelMetrics:
    @staticmethod
    def regression(y, pred):
        return pd.Series({
            "MAE": mean_absolute_error(y, pred),
            "RMSE": np.sqrt(mean_squared_error(y, pred)),
            "R2": r2_score(y, pred),
            "DA": np.mean(np.sign(y) == np.sign(pred)),
        })

    @staticmethod
    def classification(y, p, threshold=0.5):
        y = np.asarray(y).astype(int)
        p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
        pred = (p >= threshold).astype(int)
        out = {
            "LogLoss": log_loss(y, p),
            "Brier": brier_score_loss(y, p),
            "Accuracy": accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "EventRate": y.mean(),
        }
        try:
            out["AUC"] = roc_auc_score(y, p)
        except Exception:
            out["AUC"] = np.nan
        try:
            out["PR_AUC"] = average_precision_score(y, p)
        except Exception:
            out["PR_AUC"] = np.nan
        q = pd.qcut(pd.Series(p), 10, labels=False, duplicates="drop")
        top = y[q == q.max()].mean() if q.notna().any() else np.nan
        out["TopDecileLift"] = top / y.mean() if y.mean() > 0 else np.nan
        return pd.Series(out)

    @staticmethod
    def strategy(r, periods=252):
        r = pd.Series(r).dropna()
        wealth = (1 + r).cumprod()
        peak = wealth.cummax()
        dd = wealth / peak - 1
        ann_ret = (
            wealth.iloc[-1] ** (periods / len(r)) - 1 if len(r) > 2 else np.nan
        )
        ann_vol = r.std() * np.sqrt(periods)
        down_vol = r[r < 0].std() * np.sqrt(periods)
        return pd.Series({
            "Ann_Return": ann_ret,
            "Ann_Vol": ann_vol,
            "Sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
            "Sortino": ann_ret / down_vol if down_vol > 0 else np.nan,
            "MaxDD": dd.min(),
            "CVaR_5pct": r[r <= r.quantile(0.05)].mean(),
            "HitRate": (r > 0).mean(),
        })
