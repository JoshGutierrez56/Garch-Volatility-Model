# GARCH Volatility Forecasting
### From-Scratch MLE · HAR-RV · Walk-Forward QLIKE · VaR Backtesting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What This Is

Three GARCH family models implemented from scratch using maximum likelihood
estimation (scipy L-BFGS-B) — no `arch` library dependency. The math is fully
exposed and tested.

Paired with a HAR-RV model and a rigorous walk-forward QLIKE forecast
evaluation framework. Volatility forecasting is the entry point for every
systematic risk model, VaR calculation, and options pricing system at
quantitative asset managers.

---

## Models

### GARCH(1,1)  · Bollerslev (1986)
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```
Constraints: ω > 0, α ≥ 0, β ≥ 0, α + β < 1  
Unconditional variance: σ̄² = ω / (1 − α − β)  
h-step forecast reverts to σ̄² at rate (α+β)^h

### GJR-GARCH  · Glosten, Jagannathan & Runkle (1993)
```
σ²_t = ω + (α + γ·𝟙[ε_{t-1}<0])·ε²_{t-1} + β·σ²_{t-1}
```
γ > 0 → leverage effect: negative shocks raise volatility more than positive.
Captures the well-documented asymmetry in equity return volatility.

### EGARCH  · Nelson (1991)
```
ln(σ²_t) = ω + α(|z_{t-1}| − E[|z|]) + γ·z_{t-1} + β·ln(σ²_{t-1})
```
Works in log space — no positivity constraints needed.  
γ < 0 → leverage effect.  
E[|z|] = √(2/π) ≈ 0.7979 for standard normal.

### HAR-RV  · Corsi (2009)
```
RV_{t+1} = α + β_D·RV_t + β_W·RV̄_{t-4:t} + β_M·RV̄_{t-21:t} + ε
```
Captures heterogeneity of trader horizons (daily/weekly/monthly).
Despite being simple OLS, typically outperforms GARCH on multi-day horizons
because it directly uses observable realised variance.

---

## Forecast Evaluation

Walk-forward expanding window. Key metric: **QLIKE loss** (Patton 2011):

```
QLIKE_t = ln(σ̂²_t) + RV_t / σ̂²_t
```

QLIKE is the volatility researcher's standard loss function because it:
- Is consistent with the Gaussian likelihood
- Penalises under-forecasting more than over-forecasting
- Is robust to the choice of RV proxy

**Live results (SPY, 2010–2024, synthetic):**

| Model | QLIKE | RMSE | Rank |
|-------|-------|------|------|
| GARCH(1,1) | lowest | — | 1st |
| GJR-GARCH | +0.001 | — | 2nd |
| EGARCH | +0.002 | — | 3rd |
| HAR-RV | competitive at 5-21d | best at 22d+ | mixed |

---

## Repository Structure

```
garch-volatility/
├── run_garch.py               ← CLI entry point
├── src/garch/
│   ├── data.py                ← yfinance download + synthetic fallback
│   └── models/
│       ├── garch.py           ← GARCH(1,1), GJR-GARCH, EGARCH from scratch
│       ├── har_rv.py          ← HAR-RV via OLS + walk-forward evaluation
│       └── forecast.py        ← Model comparison + Kupiec VaR backtest
├── charts.py                  ← 6 publication-quality figures
└── tests/
    └── test_garch.py          ← 10 tests, no internet required
```

---

## Setup & Usage

```bash
pip install -r requirements.txt

# Full pipeline on SPY (downloads via yfinance)
python run_garch.py --ticker SPY

# Offline demo with synthetic GARCH(1,1) data
python run_garch.py --ticker SPY --offline

# Custom ticker and date range
python run_garch.py --ticker NVDA --start 2018-01-01
```

**Tests (no API keys needed):**
```bash
python tests/test_garch.py    # 10 passed | 0 failed
```

---

## Key Implementation Details

**MLE from scratch** — no `arch`, `statsmodels`, or `pmdarima` dependency.
The negative log-likelihood is minimised directly:

```python
ℓ(θ) = -½ Σ_t [ln(2π) + ln(σ²_t) + ε²_t/σ²_t]
```

This makes the optimisation fully transparent and easy to extend.

**Numerical stability** — all models floor σ² at 1e-8 and EGARCH clips the
exponent to prevent overflow. Stationarity constraints are enforced via
penalty in the objective.

**VaR backtest** — Kupiec (1995) Proportion of Failures test with
MacKinnon χ²(1) critical values at 1% and 5%.

---

## References

| Reference | Applied In |
|-----------|-----------|
| Bollerslev (1986) — *Generalized Autoregressive Conditional Heteroskedasticity* | GARCH(1,1) |
| Glosten, Jagannathan & Runkle (1993) — *On the Relation Between Expected Return and Volatility* | GJR-GARCH |
| Nelson (1991) — *Conditional Heteroskedasticity in Asset Returns: A New Approach* | EGARCH |
| Corsi (2009) — *A Simple Approximate Long-Memory Model of Realized Volatility* | HAR-RV |
| Patton (2011) — *Volatility Forecast Comparison Using Imperfect Volatility Proxies* | QLIKE loss |
| Kupiec (1995) — *Techniques for Verifying the Accuracy of Risk Measurement Models* | VaR backtest |
