"""
config.py
---------
Central configuration dataclass for the FII-DII market ML pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    start_date: str = "2015-01-01"
    end_date: str = "2026-05-01"
    data_dir: str = "/content/st451_fii_dii_data"
    horizon: int = 5
    test_size: float = 0.25
    downside_quantile: float = 0.20
    action_z: float = 0.60
    random_state: int = 42
    allow_demo_if_fiidii_missing: bool = True
    min_fiidii_dates: int = 250
    pymc_max_rows: int = 6000
    pymc_draws: int = 1000
    pymc_tune: int = 1500
    pymc_chains: int = 4
    transaction_cost: float = 0.0005
    rebalance_step: int = 5
    nse_index_names: dict = field(default_factory=lambda: {
        "NIFTY50": "NIFTY 50",
        "BANK": "NIFTY BANK",
        "IT": "NIFTY IT",
        "FMCG": "NIFTY FMCG",
        "AUTO": "NIFTY AUTO",
        "PHARMA": "NIFTY PHARMA",
        "METAL": "NIFTY METAL",
        "REALTY": "NIFTY REALTY",
        "FIN_SERVICE": "NIFTY FINANCIAL SERVICES",
    })
    yahoo_fallback: dict = field(default_factory=lambda: {
        "NIFTY50": "^NSEI",
        "VIX": "^INDIAVIX",
        "BANK": "^NSEBANK",
        "IT": "^CNXIT",
        "FMCG": "^CNXFMCG",
        "AUTO": "^CNXAUTO",
        "PHARMA": "^CNXPHARMA",
        "METAL": "^CNXMETAL",
        "REALTY": "^CNXREALTY",
        "FIN_SERVICE": "NIFTY_FIN_SERVICE.NS",
    })
    defensive_sectors: tuple = ("FMCG", "PHARMA", "IT")
    cyclical_sectors: tuple = ("BANK", "AUTO", "METAL", "REALTY", "FIN_SERVICE")


def get_config() -> Config:
    cfg = Config()
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    return cfg
