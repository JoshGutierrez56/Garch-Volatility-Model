"""Rebuild an illustrative news-impact comparison from the project's original equations.

These parameters are specified, not estimated. No return or forecast-accuracy claim.
Uses the same equations as src/garch/charts.py::fig2_news_impact in the legacy ZIP.
"""
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def variance(shock: float, asymmetric: bool) -> float:
    prior_variance = 0.20 ** 2 / 252
    omega = 0.05 * prior_variance
    alpha, gamma, beta = (0.05, 0.10, 0.85) if asymmetric else (0.10, 0.0, 0.85)
    return omega + (alpha + gamma * (shock < 0)) * shock ** 2 + beta * prior_variance


def main():
    # Independent hand arithmetic: base = .9*(.04/252), shock +/-2%.
    base = .9 * .04 / 252
    assert math.isclose(variance(.02, False), base + .10 * .0004, abs_tol=1e-15)
    assert math.isclose(variance(-.02, False), variance(.02, False), abs_tol=1e-15)
    assert math.isclose(variance(.02, True), base + .05 * .0004, abs_tol=1e-15)
    assert math.isclose(variance(-.02, True), base + .15 * .0004, abs_tol=1e-15)
    rows = [{"shock_pct": i / 4, "garch_vol_pct": math.sqrt(variance(i/400, False)*252)*100,
             "gjr_vol_pct": math.sqrt(variance(i/400, True)*252)*100} for i in range(-20, 21)]
    payload = {"scope": "Illustrative parameter comparison; no estimation or market-data result",
               "source": "garch_volatility.zip: garch-volatility/src/garch/charts.py fig2_news_impact",
               "prior_annual_volatility": 0.20, "annualization": 252,
               "garch": {"omega": .05*.04/252, "alpha": .1, "beta": .85},
               "gjr": {"omega": .05*.04/252, "alpha": .05, "gamma": .1, "beta": .85}, "rows": rows}
    output = ROOT / "docs" / "exhibit"
    output.mkdir(parents=True, exist_ok=True)
    (output / "news-impact.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, key, color in [("GARCH: symmetric", "garch_vol_pct", "#275b97"),
                             ("GJR-GARCH: asymmetric", "gjr_vol_pct", "#a34b37")]:
        ax.plot([r["shock_pct"] for r in rows], [r[key] for r in rows], label=name, color=color, linewidth=2.5)
    ax.axvline(0, color="#b5bdc7", linewidth=.8)
    ax.axhline(20, color="#8190a4", linestyle="--", linewidth=.8, label="Prior annualized volatility: 20%")
    ax.set(xlabel="Return innovation (%)", ylabel="Next-session volatility (%, annualized)",
           title="Same-sized shock. Different risk response.")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=.2)
    fig.text(.5, .015, "Illustrative parameters, not fitted market results. Annualization: 252 sessions.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(output/"news-impact.png", dpi=160)
    plt.close(fig)
    print("Four independent arithmetic checks passed; wrote 41 shock scenarios and source chart.")


if __name__ == "__main__":
    main()
