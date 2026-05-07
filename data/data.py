"""
data.py
-------
NSEDataClient  – scrapes NSE and falls back to Yahoo Finance.
DataManager    – orchestrates price and FII/DII flow loading.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import Config


class NSEDataClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base = "https://www.nseindia.com"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/121.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        })
        self.refresh()

    def refresh(self):
        try:
            self.session.get(self.base, timeout=20)
        except Exception:
            pass

    def dmy(self, x):
        return pd.to_datetime(x).strftime("%d-%m-%Y")

    def chunks(self, start, end, days=360):
        cur = pd.to_datetime(start)
        end = pd.to_datetime(end)
        while cur <= end:
            nxt = min(cur + pd.Timedelta(days=days), end)
            yield cur, nxt
            cur = nxt + pd.Timedelta(days=1)

    def get_json(self, endpoint, params=None):
        url = self.base + endpoint
        r = self.session.get(url, params=params, timeout=30)
        if r.status_code in (401, 403):
            self.refresh()
            r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def largest_table(self, obj):
        tables = []

        def walk(x):
            if isinstance(x, list):
                if len(x) and all(isinstance(i, dict) for i in x):
                    tables.append(pd.json_normalize(x))
                for i in x:
                    walk(i)
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)

        walk(obj)
        return max(tables, key=len).copy() if tables else pd.DataFrame()

    def num(self, s):
        return pd.to_numeric(
            pd.Series(s)
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False),
            errors="coerce",
        )

    def yahoo_series(self, ticker, label):
        df = yf.download(
            ticker,
            start=self.cfg.start_date,
            end=self.cfg.end_date,
            progress=False,
            auto_adjust=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={"Date": "date", "Close": label})
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", label]].dropna()

    def index_history(self, index_name, label):
        path = Path(self.cfg.data_dir) / f"index_{label}.csv"
        if path.exists():
            return pd.read_csv(path)
        frames = []
        for a, b in self.chunks(self.cfg.start_date, self.cfg.end_date):
            try:
                js = self.get_json(
                    "/api/historical/indicesHistory",
                    {"indexType": index_name, "from": self.dmy(a), "to": self.dmy(b)},
                )
                df = self.largest_table(js)
                if len(df):
                    frames.append(df)
                time.sleep(0.25)
            except Exception:
                time.sleep(0.25)
        if frames:
            out = pd.concat(frames, ignore_index=True).drop_duplicates()
            out.to_csv(path, index=False)
            return out
        ticker = self.cfg.yahoo_fallback.get(label)
        if ticker is None:
            raise RuntimeError(label)
        out = self.yahoo_series(ticker, label)
        out.to_csv(path, index=False)
        return out

    def vix_history(self):
        path = Path(self.cfg.data_dir) / "india_vix.csv"
        if path.exists():
            return pd.read_csv(path)
        frames = []
        for a, b in self.chunks(self.cfg.start_date, self.cfg.end_date):
            try:
                js = self.get_json(
                    "/api/historical/vixhistory",
                    {"from": self.dmy(a), "to": self.dmy(b)},
                )
                df = self.largest_table(js)
                if len(df):
                    frames.append(df)
                time.sleep(0.25)
            except Exception:
                time.sleep(0.25)
        if frames:
            out = pd.concat(frames, ignore_index=True).drop_duplicates()
            out.to_csv(path, index=False)
            return out
        out = self.yahoo_series(self.cfg.yahoo_fallback["VIX"], "vix_close")
        out.to_csv(path, index=False)
        return out

    def fiidii_latest(self):
        try:
            js = self.get_json("/api/fiidiiTradeReact")
            return self.largest_table(js)
        except Exception:
            return pd.DataFrame()


class DataManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)
        self.client = NSEDataClient(cfg)

    def num(self, s):
        return pd.to_numeric(
            pd.Series(s)
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("+", "", regex=False),
            errors="coerce",
        )

    def find_col(self, df, pats):
        low = {c: str(c).lower().replace(" ", "_") for c in df.columns}
        for p in pats:
            for c, l in low.items():
                if p.lower() in l:
                    return c
        return None

    def read_csv(self, name):
        path = self.data_dir / name
        return pd.read_csv(path) if path.exists() else None

    def clean_vix(self, raw):
        date_col = self.find_col(raw, ["date", "timestamp"])
        close_col = self.find_col(raw, ["vix_close", "close", "closing"])
        out = pd.DataFrame({
            "date": pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True),
            "vix_close": self.num(raw[close_col]),
        })
        return out.dropna().drop_duplicates("date").sort_values("date")

    def clean_index_raw(self, raw, label):
        date_col = self.find_col(raw, ["date", "timestamp"])
        close_col = self.find_col(
            raw, [label.lower(), "close_index_val", "closing", "close", "last"]
        )
        out = pd.DataFrame({
            "date": pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True),
            label: self.num(raw[close_col]),
        })
        return out.dropna().drop_duplicates("date").sort_values("date")

    def load_prices(self):
        local = self.read_csv("prices.csv")
        if local is not None:
            local["date"] = pd.to_datetime(local["date"], errors="coerce", dayfirst=True)
            return local.dropna(subset=["date"]).sort_values("date"), "local_prices_csv"

        tables, failures = [], []
        for label, index_name in self.cfg.nse_index_names.items():
            try:
                raw = self.client.index_history(index_name, label)
                clean = self.clean_index_raw(raw, label)
                if len(clean) > 100:
                    tables.append(clean)
                else:
                    raise RuntimeError("too_few_rows")
            except Exception as e:
                failures.append((label, str(e)))

        try:
            vix = self.clean_vix(self.client.vix_history())
        except Exception:
            vix = self.client.yahoo_series(self.cfg.yahoo_fallback["VIX"], "vix_close")

        if tables:
            prices = tables[0]
            for t in tables[1:]:
                prices = prices.merge(t, on="date", how="outer")
            prices = prices.merge(vix, on="date", how="outer")
        else:
            series = []
            for label, ticker in self.cfg.yahoo_fallback.items():
                if label != "VIX":
                    try:
                        s = self.client.yahoo_series(ticker, label)
                        if len(s) > 100:
                            series.append(s)
                    except Exception:
                        pass
            prices = series[0]
            for s in series[1:]:
                prices = prices.merge(s, on="date", how="outer")
            prices = prices.merge(vix, on="date", how="outer")

        prices = prices.sort_values("date").drop_duplicates("date")
        return prices, {"source": "nse_first_yahoo_fallback", "index_failures": failures}

    def clean_fiidii(self, raw):
        df = raw.copy()
        date_col = self.find_col(df, ["date", "timestamp", "trade_date"])
        if date_col is None:
            raise ValueError("date_missing")
        cat_col = self.find_col(df, ["category", "client", "investor"])
        net_col = self.find_col(df, ["netvalue", "net_value", "net"])
        buy_col = self.find_col(df, ["buyvalue", "buy_value", "gross_buy", "buy"])
        sell_col = self.find_col(df, ["sellvalue", "sell_value", "gross_sell", "sell"])

        if cat_col is not None and net_col is not None:
            temp = pd.DataFrame({
                "date": pd.to_datetime(df[date_col], errors="coerce", dayfirst=True),
                "category": df[cat_col].astype(str),
                "net": self.num(df[net_col]),
            })
            if buy_col is not None:
                temp["buy"] = self.num(df[buy_col])
            if sell_col is not None:
                temp["sell"] = self.num(df[sell_col])
            cat = temp["category"].str.upper()
            temp["group"] = np.select(
                [
                    cat.str.contains("DII", na=False),
                    cat.str.contains("FII|FPI", regex=True, na=False),
                ],
                ["dii", "fii"],
                default=None,
            )
            temp = temp.dropna(subset=["date", "group"])
            out = (
                temp.pivot_table(
                    index="date", columns="group", values="net", aggfunc="sum"
                )
                .rename(columns={"fii": "fii_net", "dii": "dii_net"})
                .reset_index()
            )
            if "buy" in temp:
                out = out.merge(
                    temp.pivot_table(
                        index="date", columns="group", values="buy", aggfunc="sum"
                    )
                    .rename(columns={"fii": "fii_buy", "dii": "dii_buy"})
                    .reset_index(),
                    on="date",
                    how="left",
                )
            if "sell" in temp:
                out = out.merge(
                    temp.pivot_table(
                        index="date", columns="group", values="sell", aggfunc="sum"
                    )
                    .rename(columns={"fii": "fii_sell", "dii": "dii_sell"})
                    .reset_index(),
                    on="date",
                    how="left",
                )
        else:
            low = {c: str(c).lower().replace(" ", "_") for c in df.columns}
            fii = next(
                (c for c, l in low.items() if "fii" in l and ("net" in l or "activity" in l)),
                None,
            )
            dii = next(
                (c for c, l in low.items() if "dii" in l and ("net" in l or "activity" in l)),
                None,
            )
            if fii is None or dii is None:
                raise ValueError(str(df.columns.tolist()))
            out = pd.DataFrame({
                "date": pd.to_datetime(df[date_col], errors="coerce", dayfirst=True),
                "fii_net": self.num(df[fii]),
                "dii_net": self.num(df[dii]),
            })

        return (
            out.dropna(subset=["date", "fii_net", "dii_net"])
            .drop_duplicates("date")
            .sort_values("date")
        )

    def validate_flows(self, flows):
        n = flows["date"].nunique()
        if n < self.cfg.min_fiidii_dates:
            raise RuntimeError(
                f"FII/DII has only {n} unique dates. Expected historical data."
            )
        return flows

    def demo_flows(self, prices):
        dates = pd.to_datetime(prices["date"])
        ret = (
            prices["NIFTY50"].pct_change().fillna(0).values
            if "NIFTY50" in prices
            else np.random.normal(0, 0.01, len(prices))
        )
        vix = (
            prices["vix_close"].ffill().values
            if "vix_close" in prices
            else np.repeat(18, len(prices))
        )
        rng = np.random.default_rng(self.cfg.random_state)
        stress = (vix - np.nanmean(vix)) / (np.nanstd(vix) + 1e-8)
        fii = 180000 * ret - 450 * stress + rng.normal(0, 900, len(prices))
        dii = -0.65 * fii + 250 * stress + rng.normal(0, 700, len(prices))
        return pd.DataFrame({"date": dates, "fii_net": fii, "dii_net": dii})

    def load_flows(self, prices):
        local = self.read_csv("fii_dii.csv")
        if local is not None:
            flows = self.clean_fiidii(local)
            return self.validate_flows(flows), "local_fii_dii_csv"
        latest = self.client.fiidii_latest()
        if len(latest):
            try:
                flows = self.clean_fiidii(latest)
                return self.validate_flows(flows), "nse_fii_dii_historical"
            except Exception as e:
                latest.to_csv(self.data_dir / "bad_latest_fii_dii.csv", index=False)
                if not self.cfg.allow_demo_if_fiidii_missing:
                    raise e
        if not self.cfg.allow_demo_if_fiidii_missing:
            raise RuntimeError(
                "Upload historical fii_dii.csv with columns date, fii_net, dii_net."
            )
        return self.demo_flows(prices), "demo_flows"

    def load_all(self):
        prices, price_source = self.load_prices()
        flows, flow_source = self.load_flows(prices)
        return prices, flows, {"prices": price_source, "flows": flow_source}
