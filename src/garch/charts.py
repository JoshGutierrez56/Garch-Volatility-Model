"""
charts.py  ·  garch-volatility-forecasting
===========================================
Figures
1. Conditional Volatility — GARCH vs GJR vs EGARCH overlaid on returns
2. GARCH Persistence / News Impact Curve (GJR asymmetry)
3. Model Comparison — QLIKE and RMSE bar chart
4. Walk-Forward Forecast vs Realised Variance
5. VaR Backtest — violations highlighted on return series
6. ACF of Squared Returns and Standardised Residuals
7. HAR-RV Coefficient Decomposition
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False,    "axes.spines.right": False,
    "font.size": 10,             "axes.titlesize": 12,
    "axes.titleweight": "bold",  "figure.dpi": 130,
})


def _save(fig, out: Path, name: str) -> None:
    p = out / name
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", p)


def fig1_conditional_vol(returns: np.ndarray, garch_results: dict,
                         ticker: str, out: Path) -> None:
    t = np.arange(len(returns))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Returns
    ax = axes[0]
    ax.plot(t, returns * 100, color="#888888", lw=0.6, alpha=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Daily Return (%)")
    ax.set_title(f"Figure 1: {ticker} Daily Returns", fontweight="bold")

    # Conditional volatility
    ax2 = axes[1]
    colors = {"GARCH(1,1)": "#2166ac", "GJR-GARCH": "#d6604d", "EGARCH": "#276749"}
    for name, res in garch_results.items():
        if res is None: continue
        cond_vol = np.sqrt(res.cond_var) * np.sqrt(252) * 100
        ax2.plot(t, cond_vol, color=colors.get(name, "#888"), lw=1.5, label=name)

    ax2.set_ylabel("Conditional Volatility (%, ann.)")
    ax2.set_title("Conditional Volatility — GARCH Models", fontweight="bold")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(axis="y", alpha=0.2)

    fig.suptitle(f"Figure 1: {ticker} — GARCH Conditional Volatility", fontsize=12)
    plt.tight_layout()
    _save(fig, out, "fig1_conditional_vol.png")


def fig2_news_impact(garch_results: dict, out: Path) -> None:
    eps_range = np.linspace(-0.05, 0.05, 300)
    avg_sigma2 = {name: res.cond_var.mean()
                  for name, res in garch_results.items() if res is not None}

    fig, axes = plt.subplots(1, len(garch_results), figsize=(15, 5), sharey=True)
    if len(garch_results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, garch_results.items()):
        if res is None: continue
        p = res.params
        s2_bar = avg_sigma2[name]

        if name == "GARCH(1,1)":
            nic = p["omega"] + p["alpha"] * eps_range ** 2 + p["beta"] * s2_bar
        elif name == "GJR-GARCH":
            ind = (eps_range < 0).astype(float)
            nic = p["omega"] + (p["alpha"] + p["gamma"] * ind) * eps_range ** 2 + p["beta"] * s2_bar
        else:  # EGARCH
            z = eps_range / np.sqrt(s2_bar)
            ln_s2_bar = np.log(s2_bar)
            EZ = np.sqrt(2 / np.pi)
            ln_nic = p["omega"] + p["alpha"] * (np.abs(z) - EZ) + p["gamma"] * z + p["beta"] * ln_s2_bar
            nic = np.exp(ln_nic)

        ax.plot(eps_range * 100, np.sqrt(nic) * np.sqrt(252) * 100,
                color="#2166ac", lw=2.5)
        ax.axvline(0, color="black", lw=0.7, ls="--")
        ax.set_xlabel("Innovation ε_t (%)")
        if name == list(garch_results.keys())[0]:
            ax.set_ylabel("σ_{t+1} (%, ann.)")
        ax.set_title(name, fontweight="bold")
        ax.grid(alpha=0.2)

    fig.suptitle("Figure 2: News Impact Curves\n"
                 "How shocks of different sign/magnitude affect next-period volatility",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, out, "fig2_news_impact.png")


def fig3_model_comparison(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    models = summary.index.tolist()
    x = np.arange(len(models))

    for ax, metric, color in zip(axes, ["QLIKE", "RMSE"], ["#2166ac", "#d6604d"]):
        vals = summary[metric].values
        bars = ax.bar(models, vals, color=color, alpha=0.85, edgecolor="white")
        # Highlight best (lowest)
        best_idx = np.argmin(vals)
        bars[best_idx].set_edgecolor("black"); bars[best_idx].set_linewidth(2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.01,
                    f"{val:.2e}", ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{metric} (lower = better)", fontweight="bold")
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Figure 3: Walk-Forward Forecast Evaluation\n"
                 "QLIKE = volatility-consistent loss | RMSE = root mean squared error",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, out, "fig3_model_comparison.png")


def fig4_forecast_vs_realized(wf_df: pd.DataFrame, model_name: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    t = np.arange(len(wf_df))
    ax.plot(t, np.sqrt(wf_df["actual_rv"])   * np.sqrt(252) * 100,
            color="#888888", lw=1.2, label="Realised Vol", alpha=0.8)
    ax.plot(t, np.sqrt(wf_df["forecast_var"]) * np.sqrt(252) * 100,
            color="#2166ac", lw=2, label=f"{model_name} Forecast")
    ax.set_ylabel("Annualised Volatility (%)")
    ax.set_title(f"Figure 4: {model_name} — Walk-Forward Forecast vs Realised Variance", pad=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, out, "fig4_forecast_vs_realized.png")


def fig5_var_backtest(returns: np.ndarray, var_series: np.ndarray,
                      kupiec: dict, ticker: str, out: Path) -> None:
    n      = min(len(returns), len(var_series))
    ret    = returns[-n:] * 100
    var    = var_series[-n:] * 100
    viol   = ret < -var

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ret, color="#888888", lw=0.6, alpha=0.7, label="Daily Return")
    ax.plot(-var, color="#2166ac", lw=1.8, label="99% VaR (negative)")
    ax.scatter(np.where(viol)[0], ret[viol], color="#c1121f", s=20,
               zorder=5, label=f"Violations ({kupiec['violations']})")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Return / VaR (%)")
    ax.set_title(
        f"Figure 5: {ticker} 99% VaR Backtest (GARCH)\n"
        f"Violations: {kupiec['violations']} ({kupiec['rate']:.2%} vs expected {kupiec['expected_rate']:.2%})"
        f"  |  Kupiec p-value: {kupiec['p_value']:.4f}",
        pad=10,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, out, "fig5_var_backtest.png")


def fig6_acf_diagnostics(returns: np.ndarray, std_resid: np.ndarray, out: Path) -> None:
    from scipy.signal import correlate

    def acf(x, nlags=30):
        x = x - x.mean()
        full = correlate(x, x, mode="full")
        acf_vals = full[len(x)-1:] / full[len(x)-1]
        return acf_vals[1:nlags+1]

    lags = np.arange(1, 31)
    ci   = 1.96 / np.sqrt(len(returns))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, series, label in [
        (axes[0,0], returns ** 2,      "ACF of r²_t  (before GARCH)"),
        (axes[0,1], std_resid ** 2,    "ACF of z²_t  (after GARCH — should be white noise)"),
        (axes[1,0], returns,           "ACF of r_t"),
        (axes[1,1], std_resid,         "ACF of z_t"),
    ]:
        vals = acf(series)
        ax.bar(lags, vals, color="#2166ac", alpha=0.7, width=0.6)
        ax.axhline(ci, color="#c1121f", lw=1, ls="--", label=f"±{ci:.3f} (95% CI)")
        ax.axhline(-ci, color="#c1121f", lw=1, ls="--")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Figure 6: ACF Diagnostics — GARCH Captures Volatility Clustering",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, out, "fig6_acf_diagnostics.png")


def save_all(returns, garch_results, summary, wf_results,
             har_result, var_series, kupiec, ticker, out_dir):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Best GARCH model by QLIKE
    best_model = summary["QLIKE"].idxmin()
    best_res   = garch_results.get(best_model)

    fig1_conditional_vol(returns, garch_results, ticker, fig_dir)
    fig2_news_impact(garch_results, fig_dir)
    fig3_model_comparison(summary, fig_dir)
    if best_model in wf_results and not wf_results[best_model].empty:
        fig4_forecast_vs_realized(wf_results[best_model], best_model, fig_dir)
    if best_res is not None:
        fig5_var_backtest(returns, var_series, kupiec, ticker, fig_dir)
        fig6_acf_diagnostics(returns, best_res.std_resid, fig_dir)
    logger.info("All figures saved to %s", fig_dir)
