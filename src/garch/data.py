"""
data.py  ·  garch-volatility-forecasting
=========================================
Download daily prices via yfinance and compute return series.

Realized variance proxies (from daily OHLC — no intraday data needed)
----------------------------------------------------------------------
1. Squared daily return    : RV_t = r_t²  (simplest)
2. Parkinson (1980)        : uses (High − Low) range
3. Garman-Klass (1980)     : uses O/H/L/C, unbiased, 7× more efficient

We use squared returns as the primary RV proxy for HAR-RV since they
are directly comparable to GARCH conditional variance forecasts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default universe for multi-ticker analysis ─────────────────────────
DEFAULT_TICKERS = {
    "Equity":  ["SPY", "QQQ", "IWM"],
    "Single":  ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
    "Sector":  ["XLF", "XLE", "XLK", "XLV"],
    "Crypto":  ["BTC-USD", "ETH-USD"],
}


def download(
    tickers: List[str],
    start:   str = "2010-01-01",
    end:     Optional[str] = None,
    cache:   str = "data",
) -> pd.DataFrame:
    """
    Download adjusted close prices via yfinance.

    Returns
    -------
    pd.DataFrame  — wide, rows=dates, cols=tickers (adjusted close)
    """
    cache_path = Path(cache) / f"prices_{'_'.join(sorted(tickers)[:4])}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        logger.info("Prices loaded from cache (%d rows × %d tickers)",
                    len(df), len(df.columns))
        return df

    try:
        import yfinance as yf
        logger.info("Downloading %d tickers (%s → %s)...", len(tickers), start, end or "today")
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False, threads=True)
        prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        prices.index = pd.DatetimeIndex(prices.index)
        if hasattr(prices.index, "tz") and prices.index.tz:
            prices.index = prices.index.tz_localize(None)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)
        return prices
    except Exception as exc:
        logger.warning("yfinance failed (%s) — generating synthetic data", exc)
        return synthetic_prices(tickers, start, end)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns: r_t = ln(P_t / P_{t-1})"""
    return np.log(prices / prices.shift(1)).dropna()


def squared_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Squared log returns as realized variance proxy."""
    return returns ** 2


def parkinson_rv(ohlc: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Parkinson (1980) range-based volatility estimator.
    Var = (1 / 4 ln2) × E[(ln H − ln L)²]
    Annualised variance: multiply by 252.
    """
    hl = np.log(ohlc["High"] / ohlc["Low"])
    rv = hl ** 2 / (4 * np.log(2))
    return (rv * 252).rename(f"{ticker}_Parkinson")


def synthetic_prices(
    tickers: List[str],
    start:   str,
    end:     Optional[str],
) -> pd.DataFrame:
    """Synthetic GBM prices with realistic volatility clustering."""
    rng   = np.random.default_rng(42)
    dates = pd.bdate_range(start=start, end=end or "2024-01-01")
    n, k  = len(dates), len(tickers)

    prices = np.ones((n, k)) * 100.0
    vol    = np.ones(k) * 0.015          # initial daily vol ~1.5%
    alpha, beta = 0.10, 0.85             # GARCH-like vol clustering

    for t in range(1, n):
        shock     = rng.standard_normal(k)
        ret       = shock * vol
        prices[t] = prices[t-1] * np.exp(ret)
        vol       = np.sqrt(np.maximum(
            1e-6,
            0.0002 + alpha * (shock * vol) ** 2 + beta * vol ** 2
        ))

    return pd.DataFrame(prices, index=dates, columns=tickers)
