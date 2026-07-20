"""
garch.py  ·  garch-volatility-forecasting
==========================================
GARCH family models implemented from scratch via MLE.

Three models
------------

GARCH(1,1)   Bollerslev (1986)
────────────────────────────────
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Constraints: ω > 0, α ≥ 0, β ≥ 0, α + β < 1 (stationarity)
Unconditional variance: σ̄² = ω / (1 − α − β)

GJR-GARCH    Glosten, Jagannathan & Runkle (1993)
──────────────────────────────────────────────────
σ²_t = ω + (α + γ·I_{t-1})·ε²_{t-1} + β·σ²_{t-1}

where I_{t-1} = 1 if ε_{t-1} < 0 (bad news)
Leverage effect: γ > 0 → negative shocks increase volatility more
Constraints: ω > 0, α ≥ 0, α + γ ≥ 0, β ≥ 0, α + γ/2 + β < 1

EGARCH       Nelson (1991)
───────────────────────────
ln(σ²_t) = ω + α·(|z_{t-1}| − E[|z|]) + γ·z_{t-1} + β·ln(σ²_{t-1})

where z_{t-1} = ε_{t-1}/σ_{t-1}
Works in log space so no positivity constraints needed.
γ < 0 → leverage effect (negative returns increase volatility)

Log-likelihood (Gaussian)
──────────────────────────
ℓ(θ) = −½ Σ_t [ln(2π) + ln(σ²_t) + ε²_t/σ²_t]

Estimation via scipy.optimize.minimize with L-BFGS-B.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


@dataclass
class GARCHResult:
    """Fitted GARCH model output."""
    model:         str
    params:        Dict[str, float]
    log_likelihood: float
    aic:           float
    bic:           float
    cond_var:      np.ndarray    # σ²_t series
    std_resid:     np.ndarray    # z_t = ε_t / σ_t
    converged:     bool
    n_obs:         int


# ── Shared helpers ─────────────────────────────────────────────────────────

def _gaussian_ll(eps: np.ndarray, sigma2: np.ndarray) -> float:
    """Gaussian log-likelihood given residuals and conditional variances."""
    ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)
    return float(ll)


def _aic_bic(ll: float, k: int, n: int) -> Tuple[float, float]:
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)
    return aic, bic


# ── GARCH(1,1) ───────────────────────────────────────────────────────────────

class GARCH11:
    """GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}"""

    def __init__(self) -> None:
        self.result: Optional[GARCHResult] = None

    def _filter(self, params: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """Compute conditional variance series via GARCH recursion."""
        omega, alpha, beta = params
        n      = len(eps)
        sigma2 = np.empty(n)
        sigma2[0] = np.var(eps)                     # initialise at sample variance
        for t in range(1, n):
            sigma2[t] = omega + alpha * eps[t-1] ** 2 + beta * sigma2[t-1]
            sigma2[t] = max(sigma2[t], 1e-8)        # floor to prevent numerical issues
        return sigma2

    def _neg_ll(self, params: np.ndarray, eps: np.ndarray) -> float:
        omega, alpha, beta = params
        # Positivity + stationarity constraints via penalty
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        sigma2 = self._filter(params, eps)
        return -_gaussian_ll(eps, sigma2)

    def fit(self, returns: np.ndarray) -> GARCHResult:
        """Fit GARCH(1,1) by MLE."""
        eps = returns - returns.mean()
        var0 = np.var(eps)

        # Starting values: ω = 0.05·σ̄², α = 0.10, β = 0.85
        p0     = np.array([0.05 * var0, 0.10, 0.85])
        bounds = [(1e-8, None), (1e-8, 0.999), (1e-8, 0.999)]

        res = minimize(self._neg_ll, p0, args=(eps,), method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 2000, "ftol": 1e-12})

        omega, alpha, beta = res.x
        sigma2 = self._filter(res.x, eps)
        ll     = -res.fun
        aic, bic = _aic_bic(ll, 3, len(eps))

        self.result = GARCHResult(
            model="GARCH(1,1)",
            params={"omega": omega, "alpha": alpha, "beta": beta,
                    "persistence": alpha + beta,
                    "uncond_var": omega / max(1 - alpha - beta, 1e-8)},
            log_likelihood=ll, aic=aic, bic=bic,
            cond_var=sigma2, std_resid=eps / np.sqrt(sigma2),
            converged=res.success, n_obs=len(eps),
        )
        self._params = res.x
        self._eps    = eps
        logger.info("GARCH(1,1): ω=%.2e α=%.4f β=%.4f persist=%.4f LL=%.2f",
                    omega, alpha, beta, alpha+beta, ll)
        return self.result

    def forecast(self, h: int = 1) -> np.ndarray:
        """
        h-step ahead variance forecast.

        For GARCH(1,1): σ²_{T+h} = σ̄² + (α+β)^h × (σ²_T − σ̄²)
        Reverts to unconditional variance as h → ∞.
        """
        if self.result is None:
            raise RuntimeError("Model not fitted — call fit() first")
        omega, alpha, beta = self._params
        persist  = alpha + beta
        uncond   = omega / max(1 - persist, 1e-8)
        sigma2_T = self.result.cond_var[-1]
        return np.array([
            uncond + persist ** i * (sigma2_T - uncond) for i in range(1, h + 1)
        ])


# ── GJR-GARCH ────────────────────────────────────────────────────────────────

class GJRGARCH:
    """
    GJR-GARCH: σ²_t = ω + (α + γ·I_{t-1})·ε²_{t-1} + β·σ²_{t-1}
    Captures the leverage effect: negative shocks raise vol more than positive.
    """

    def __init__(self) -> None:
        self.result: Optional[GARCHResult] = None

    def _filter(self, params: np.ndarray, eps: np.ndarray) -> np.ndarray:
        omega, alpha, gamma, beta = params
        n      = len(eps)
        sigma2 = np.empty(n)
        sigma2[0] = np.var(eps)
        for t in range(1, n):
            ind        = 1.0 if eps[t-1] < 0 else 0.0
            sigma2[t]  = (omega
                          + (alpha + gamma * ind) * eps[t-1] ** 2
                          + beta * sigma2[t-1])
            sigma2[t]  = max(sigma2[t], 1e-8)
        return sigma2

    def _neg_ll(self, params: np.ndarray, eps: np.ndarray) -> float:
        omega, alpha, gamma, beta = params
        if (omega <= 0 or alpha < 0 or alpha + gamma < 0
                or beta < 0 or alpha + gamma / 2 + beta >= 1):
            return 1e10
        sigma2 = self._filter(params, eps)
        return -_gaussian_ll(eps, sigma2)

    def fit(self, returns: np.ndarray) -> GARCHResult:
        eps = returns - returns.mean()
        var0 = np.var(eps)
        p0   = np.array([0.05 * var0, 0.05, 0.10, 0.85])
        bounds = [(1e-8, None), (1e-8, 0.999), (-0.999, 0.999), (1e-8, 0.999)]

        res = minimize(self._neg_ll, p0, args=(eps,), method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 2000, "ftol": 1e-12})

        omega, alpha, gamma, beta = res.x
        sigma2 = self._filter(res.x, eps)
        ll     = -res.fun
        aic, bic = _aic_bic(ll, 4, len(eps))

        self.result = GARCHResult(
            model="GJR-GARCH",
            params={"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
                    "leverage_effect": gamma > 0,
                    "persistence": alpha + gamma / 2 + beta},
            log_likelihood=ll, aic=aic, bic=bic,
            cond_var=sigma2, std_resid=eps / np.sqrt(sigma2),
            converged=res.success, n_obs=len(eps),
        )
        self._params = res.x
        self._eps    = eps
        logger.info("GJR-GARCH: ω=%.2e α=%.4f γ=%.4f β=%.4f leverage=%s LL=%.2f",
                    omega, alpha, gamma, beta, gamma > 0, ll)
        return self.result

    def forecast(self, h: int = 1) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("Call fit() first")
        omega, alpha, gamma, beta = self._params
        persist  = alpha + gamma / 2 + beta
        uncond   = omega / max(1 - persist, 1e-8)
        sigma2_T = self.result.cond_var[-1]
        return np.array([
            uncond + persist ** i * (sigma2_T - uncond) for i in range(1, h + 1)
        ])


# ── EGARCH ────────────────────────────────────────────────────────────────────

class EGARCH:
    """
    EGARCH: ln(σ²_t) = ω + α(|z_{t-1}| − E[|z|]) + γ·z_{t-1} + β·ln(σ²_{t-1})
    Works in log space — no positivity constraints needed.
    γ < 0 captures the leverage effect.
    E[|z|] = √(2/π) ≈ 0.7979 for standard normal z.
    """

    EZ = np.sqrt(2 / np.pi)   # E[|z|] for standard normal

    def __init__(self) -> None:
        self.result: Optional[GARCHResult] = None

    def _filter(self, params: np.ndarray, eps: np.ndarray) -> np.ndarray:
        omega, alpha, gamma, beta = params
        n       = len(eps)
        ln_sig2 = np.empty(n)
        ln_sig2[0] = np.log(max(np.var(eps), 1e-8))
        for t in range(1, n):
            previous = np.clip(ln_sig2[t-1], -30.0, 30.0)
            z_prev     = eps[t-1] / max(np.exp(0.5 * previous), 1e-8)
            ln_sig2[t] = (omega
                          + alpha * (abs(z_prev) - self.EZ)
                          + gamma * z_prev
                          + beta  * previous)
            ln_sig2[t] = np.clip(ln_sig2[t], -30.0, 30.0)
        return np.exp(np.clip(ln_sig2, -30.0, 30.0))

    def _neg_ll(self, params: np.ndarray, eps: np.ndarray) -> float:
        _, _, _, beta = params
        if abs(beta) >= 1:           # stationarity: |β| < 1
            return 1e10
        sigma2 = self._filter(params, eps)
        if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0):
            return 1e10
        return -_gaussian_ll(eps, sigma2)

    def fit(self, returns: np.ndarray) -> GARCHResult:
        eps  = returns - returns.mean()
        var0 = np.var(eps)
        p0   = np.array([0.01 * np.log(var0), 0.10, -0.05, 0.90])
        bounds = [(-10, 10), (-2, 2), (-2, 2), (-0.999, 0.999)]

        res = minimize(self._neg_ll, p0, args=(eps,), method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 3000, "ftol": 1e-12})

        omega, alpha, gamma, beta = res.x
        sigma2 = self._filter(res.x, eps)
        ll     = -res.fun
        aic, bic = _aic_bic(ll, 4, len(eps))

        self.result = GARCHResult(
            model="EGARCH",
            params={"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
                    "leverage_effect": gamma < 0},
            log_likelihood=ll, aic=aic, bic=bic,
            cond_var=sigma2, std_resid=eps / np.sqrt(sigma2),
            converged=res.success, n_obs=len(eps),
        )
        self._params = res.x
        self._eps    = eps
        logger.info("EGARCH: ω=%.4f α=%.4f γ=%.4f β=%.4f leverage=%s LL=%.2f",
                    omega, alpha, gamma, beta, gamma < 0, ll)
        return self.result


# ── Model selection ─────────────────────────────────────────────────────────

def fit_all(returns: np.ndarray) -> Dict[str, GARCHResult]:
    """Fit all three GARCH models and return a dict of results."""
    results = {}
    for name, cls in [("GARCH(1,1)", GARCH11), ("GJR-GARCH", GJRGARCH), ("EGARCH", EGARCH)]:
        try:
            results[name] = cls().fit(returns)
        except Exception as exc:
            logger.error("%s fitting failed: %s", name, exc)
    return results


def compare_models(results: Dict[str, GARCHResult]) -> "pd.DataFrame":
    import pandas as pd
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":      name,
            "Log-Lik":    round(r.log_likelihood, 2),
            "AIC":        round(r.aic, 2),
            "BIC":        round(r.bic, 2),
            "Converged":  r.converged,
            "N Obs":      r.n_obs,
        })
    df = pd.DataFrame(rows).set_index("Model")
    df["Best AIC"] = df["AIC"] == df["AIC"].min()
    df["Best BIC"] = df["BIC"] == df["BIC"].min()
    return df
