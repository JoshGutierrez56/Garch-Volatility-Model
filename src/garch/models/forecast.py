"""
forecast.py  ·  garch-volatility-forecasting
=============================================
Walk-forward forecast evaluation and Value-at-Risk backtesting.

Forecast evaluation metrics
---------------------------
MSE   : Mean Squared Error (penalises large errors heavily)
MAE   : Mean Absolute Error (robust to outliers)
RMSE  : Root MSE (same units as volatility)
QLIKE : Quasi-likelihood loss — asymmetric, penalises under-forecasting
        QLIKE = ln(σ̂²) + RV/σ̂²  (the volatility forecaster's standard)

QLIKE is the dominant loss function in academic volatility research
because it is consistent with the Gaussian likelihood and is invariant
to scale transformations.

Reference: Patton (2011) "Volatility forecast comparison using imperfect
           volatility proxies." Journal of Econometrics.

VaR Backtesting (Kupiec 1995 / Christoffersen 1998)
----------------------------------------------------
VaR_α_t = −σ_t × Φ^{-1}(α)   (for long position, 99% VaR with α=0.01)

Kupiec test: H0 that the violation rate equals α
Christoffersen test: H0 that violations are i.i.d. (no clustering)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .garch  import GARCH11, GJRGARCH, EGARCH, GARCHResult
from .har_rv import HARRV

logger = logging.getLogger(__name__)


def walk_forward_garch(
    returns:    np.ndarray,
    train_size: int = 252,
    horizon:    int = 1,
    evaluation_step: int = 1,
    model_cls         = None,
) -> pd.DataFrame:
    """
    Rolling walk-forward GARCH vol forecast.

    Parameters
    ----------
    returns    : daily log return series
    train_size : minimum training window
    horizon    : steps ahead to forecast (1 = next day)
    model_cls  : GARCH11 | GJRGARCH | EGARCH (default: GARCH11)

    Returns
    -------
    pd.DataFrame  — date, actual_rv, forecast_var, errors, qlike
    """
    if model_cls is None:
        model_cls = GARCH11

    rows = []
    n    = len(returns)

    if evaluation_step < 1:
        raise ValueError("evaluation_step must be at least 1")

    for t in range(train_size, n - horizon, evaluation_step):
        train = returns[:t]
        model = model_cls()
        try:
            result = model.fit(train)
        except Exception:
            continue

        fcast   = model.forecast(horizon)[horizon - 1]   # h-step ahead variance
        actual  = returns[t:t+horizon].var()              # realised variance
        err     = actual - fcast
        qlike   = np.log(max(fcast, 1e-12)) + actual / max(fcast, 1e-12)

        rows.append({
            "t":           t,
            "forecast_var": fcast,
            "actual_rv":   actual,
            "error":       err,
            "sq_error":    err ** 2,
            "abs_error":   abs(err),
            "qlike":       qlike,
        })

    return pd.DataFrame(rows)


def compare_forecasts(
    returns:    np.ndarray,
    train_size: int = 252,
    evaluation_step: int = 1,
) -> pd.DataFrame:
    """
    Run walk-forward evaluation for GARCH(1,1), GJR-GARCH, EGARCH, HAR-RV.
    Return a summary table of MSE, MAE, RMSE, QLIKE per model.
    """
    rv_series = pd.Series(returns ** 2)
    results   = {}

    for name, cls in [("GARCH(1,1)", GARCH11), ("GJR-GARCH", GJRGARCH), ("EGARCH", EGARCH)]:
        logger.info("Walk-forward %s...", name)
        try:
            df = walk_forward_garch(
                returns,
                train_size,
                evaluation_step=evaluation_step,
                model_cls=cls,
            )
            results[name] = df
        except Exception as exc:
            logger.warning("%s walk-forward failed: %s", name, exc)

    # HAR-RV
    logger.info("Walk-forward HAR-RV...")
    try:
        har_df = HARRV(rv_series).fit()
        # Use the walk-forward helper from har_rv module
        from .har_rv import walk_forward as har_wf
        har_wf_df = har_wf(rv_series, train_size=train_size, horizon=1)
        har_wf_df = har_wf_df.iloc[::evaluation_step].copy()
        # Reformat to match GARCH output
        har_garch_fmt = pd.DataFrame({
            "forecast_var": har_wf_df["forecast"],
            "actual_rv":    har_wf_df["actual"],
            "sq_error":     har_wf_df["sq_error"],
            "abs_error":    har_wf_df["error"].abs(),
            "qlike":        har_wf_df["qlike"],
        })
        results["HAR-RV"] = har_garch_fmt
    except Exception as exc:
        logger.warning("HAR-RV walk-forward failed: %s", exc)

    # Summary stats
    rows = []
    for name, df in results.items():
        n = len(df)
        rows.append({
            "Model":  name,
            "MSE":    df["sq_error"].mean(),
            "MAE":    df["abs_error"].mean(),
            "RMSE":   np.sqrt(df["sq_error"].mean()),
            "QLIKE":  df["qlike"].mean(),
            "N Obs":  n,
        })

    summary = pd.DataFrame(rows).set_index("Model")
    # Rank by QLIKE (lower = better)
    for col in ["MSE", "MAE", "RMSE", "QLIKE"]:
        summary[f"{col} Rank"] = summary[col].rank().astype(int)

    logger.info("\n%s", summary[["MSE","MAE","RMSE","QLIKE"]].round(8).to_string())
    return summary, results


# ── VaR backtesting ──────────────────────────────────────────────────────────

def compute_var(
    cond_std: np.ndarray,
    alpha:    float = 0.01,
) -> np.ndarray:
    """
    1-day Value-at-Risk at confidence level (1 − α).
    VaR_α = −σ_t × Φ^{-1}(α)

    For a $1 long position: positive VaR means expected maximum loss.
    """
    return cond_std * stats.norm.ppf(1 - alpha)


def kupiec_test(
    returns:    np.ndarray,
    var_series: np.ndarray,
    alpha:      float = 0.01,
) -> Dict:
    """
    Kupiec (1995) Proportion of Failures (POF) test.

    H0: violation rate = α
    Violations: r_t < −VaR_t

    LR_POF = −2 ln[(α^x (1−α)^(T−x)) / (p̂^x (1−p̂)^(T−x))]
    Distributed χ²(1) under H0.
    """
    violations = (returns < -var_series).astype(int)
    T   = len(violations)
    x   = violations.sum()
    p   = x / T

    if x == 0 or x == T:
        return {"violations": x, "rate": p, "expected_rate": alpha,
                "lr_stat": np.nan, "p_value": np.nan, "reject_h0": False}

    lr  = -2 * (x * np.log(alpha / p) + (T - x) * np.log((1 - alpha) / (1 - p)))
    pval = 1 - stats.chi2.cdf(lr, df=1)

    return {
        "violations":    int(x),
        "rate":          round(p, 5),
        "expected_rate": alpha,
        "lr_stat":       round(lr, 4),
        "p_value":       round(pval, 4),
        "reject_h0":     pval < 0.05,
    }
