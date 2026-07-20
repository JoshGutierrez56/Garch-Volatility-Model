"""
har_rv.py  ·  garch-volatility-forecasting
===========================================
Heterogeneous Autoregressive Realized Variance (HAR-RV) model.
Corsi (2009) "A Simple Approximate Long-Memory Model of Realized Volatility"

Model Specification
-------------------
RV_{t+1} = α + β_D·RV_t + β_W·RV̄_{t-4:t} + β_M·RV̄_{t-21:t} + ε_{t+1}

Where:
    RV_t     = daily realized variance
    RV̄_{w}  = 5-day (weekly) average of past RVs
    RV̄_{m}  = 22-day (monthly) average of past RVs

Motivation
----------
HAR-RV captures the "heterogeneity of traders" across different horizons:
- Daily traders respond to yesterday's news
- Weekly traders respond to the past week's activity
- Monthly traders respond to the past month's regime

Despite being a simple OLS regression, HAR-RV typically outperforms
GARCH on multi-day volatility forecasts because it directly uses
observable realized variance rather than implied variance from returns.

Free data: we use squared daily log returns as the RV proxy.
For production, use 5-minute returns (free via yfinance 1m data).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class HARResult:
    """Fitted HAR-RV model output."""
    alpha:   float
    beta_D:  float
    beta_W:  float
    beta_M:  float
    r2:      float
    adj_r2:  float
    rmse:    float
    t_stats: dict
    p_vals:  dict
    n_obs:   int
    fitted:  np.ndarray
    resid:   np.ndarray


class HARRV:
    """
    HAR-RV model via OLS.

    Parameters
    ----------
    rv_series : pd.Series  — daily realized variance (or squared returns)
    horizon   : int        — forecast horizon in days (default 1)
    """

    def __init__(self, rv_series: pd.Series, horizon: int = 1) -> None:
        self.rv      = rv_series.values.astype(float)
        self.dates   = rv_series.index
        self.horizon = horizon
        self.result: Optional[HARResult] = None
        self._build_features()

    def _build_features(self) -> None:
        """Construct daily, weekly (5d avg), monthly (22d avg) RV features."""
        n  = len(self.rv)
        rv = self.rv

        rv_d = rv                                             # daily
        rv_w = pd.Series(rv).rolling(5,  min_periods=5).mean().values
        rv_m = pd.Series(rv).rolling(22, min_periods=22).mean().values

        # Target: forward h-day average RV
        rv_fwd = pd.Series(rv).rolling(self.horizon).mean().shift(-self.horizon).values

        # Align: we need at least 22 days of history + horizon
        start = 22
        end   = n - self.horizon

        self.X = np.column_stack([
            np.ones(end - start),
            rv_d[start:end],
            rv_w[start:end],
            rv_m[start:end],
        ])
        self.y     = rv_fwd[start:end]
        self.dates_fit = self.dates[start:end]

        # Drop NaNs
        mask     = ~(np.isnan(self.y) | np.any(np.isnan(self.X), axis=1))
        self.X   = self.X[mask]
        self.y   = self.y[mask]
        self.dates_fit = self.dates_fit[mask]

    def fit(self) -> HARResult:
        """Fit HAR-RV by OLS. Returns HARResult."""
        X, y = self.X, self.y
        n, k = X.shape

        # OLS: β = (XᵀX)⁻¹Xᵀy
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            logger.error("HAR-RV OLS failed — singular matrix")
            raise

        fitted = X @ coef
        resid  = y - fitted
        sse    = float(resid @ resid)
        sst    = float(((y - y.mean()) ** 2).sum())

        r2     = 1 - sse / sst if sst > 0 else 0.0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
        rmse   = float(np.sqrt(sse / n))

        # Standard errors via HC0 heteroskedasticity-robust covariance
        s2     = sse / (n - k)
        XtX_inv = np.linalg.inv(X.T @ X + 1e-12 * np.eye(k))
        se     = np.sqrt(np.diag(s2 * XtX_inv))
        t_vals = coef / se
        p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df=n - k))

        alpha, beta_d, beta_w, beta_m = coef
        names  = ["alpha", "beta_D", "beta_W", "beta_M"]

        self.result = HARResult(
            alpha=alpha, beta_D=beta_d, beta_W=beta_w, beta_M=beta_m,
            r2=r2, adj_r2=adj_r2, rmse=rmse,
            t_stats=dict(zip(names, t_vals)),
            p_vals=dict(zip(names, p_vals)),
            n_obs=n, fitted=fitted, resid=resid,
        )
        self._coef = coef
        logger.info("HAR-RV: α=%.2e β_D=%.4f β_W=%.4f β_M=%.4f R²=%.4f",
                    alpha, beta_d, beta_w, beta_m, r2)
        return self.result

    def forecast(self, h: int = 1) -> float:
        """
        Forecast next-h-day average realized variance.
        Uses last observed RV_d, RV_w, RV_m as features.
        """
        if self.result is None:
            self.fit()
        rv = self.rv
        rv_d = rv[-1]
        rv_w = rv[-5:].mean()
        rv_m = rv[-22:].mean()
        x_new = np.array([1.0, rv_d, rv_w, rv_m])
        return float(x_new @ self._coef)


# ── Walk-forward forecast evaluation ────────────────────────────────────────

def walk_forward(
    rv_series:  pd.Series,
    train_size: int   = 252,
    horizon:    int   = 1,
) -> pd.DataFrame:
    """
    Walk-forward HAR-RV forecast evaluation (expanding window).

    Returns
    -------
    pd.DataFrame  — date, actual, forecast, error, sq_error, qlike
    """
    rows = []
    n    = len(rv_series)
    rv   = rv_series.values.astype(float)

    for t in range(train_size + 22, n - horizon):
        train = pd.Series(rv[:t], index=rv_series.index[:t])
        model = HARRV(train, horizon=horizon)
        model.fit()
        forecast = model.forecast(horizon)

        actual = rv[t: t + horizon].mean()
        err    = actual - forecast
        qlike  = np.log(max(forecast, 1e-12)) + actual / max(forecast, 1e-12)  # QLIKE loss

        rows.append({
            "date":     rv_series.index[t],
            "actual":   actual,
            "forecast": forecast,
            "error":    err,
            "sq_error": err ** 2,
            "qlike":    qlike,
        })

    return pd.DataFrame(rows).set_index("date")
