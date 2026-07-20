#!/usr/bin/env python3
"""
run_garch.py  ·  garch-volatility-forecasting
==============================================
GARCH Volatility Forecasting Pipeline.

Usage
-----
    python run_garch.py --ticker SPY
    python run_garch.py --ticker AAPL --start 2015-01-01
    python run_garch.py --offline          # synthetic data, no internet
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger("garch_pipeline")
OUT = Path("outputs")


def _configure_console() -> None:
    """Keep Unicode status output from crashing on legacy Windows consoles."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def main(args):
    _configure_console()
    sys.path.insert(0, "src")
    from garch.data         import download, log_returns, synthetic_prices
    from garch.models.garch import fit_all, compare_models
    from garch.models.forecast import compare_forecasts, compute_var, kupiec_test
    from garch.charts       import save_all

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)

    logger.info("="*60)
    logger.info("  GARCH Volatility Forecasting  |  ticker=%s", args.ticker)
    logger.info("="*60)

    # ── Data ──────────────────────────────────────────────────────────
    if args.offline:
        prices = synthetic_prices([args.ticker], args.start, args.end)
    else:
        prices = download([args.ticker], start=args.start, end=args.end)

    rets = log_returns(prices)[args.ticker].dropna().values
    logger.info("Returns: %d observations", len(rets))

    # ── Fit all GARCH models ──────────────────────────────────────────
    logger.info("Fitting GARCH models (MLE)...")
    results = fit_all(rets)
    model_table = compare_models(results)

    print("\n┌─ Model Comparison ──────────────────────────────────────")
    print(model_table.to_string())
    print("└─────────────────────────────────────────────────────────\n")

    # ── Walk-forward forecast evaluation ──────────────────────────────
    logger.info("Running walk-forward evaluation...")
    summary, wf_results = compare_forecasts(
        rets, train_size=args.train_size, evaluation_step=args.evaluation_step
    )

    print("\n┌─ Walk-Forward Forecast Metrics ────────────────────────")
    print(summary[["MSE","MAE","RMSE","QLIKE"]].round(8).to_string())
    print("└─────────────────────────────────────────────────────────\n")

    # ── VaR backtest ──────────────────────────────────────────────────
    best_model = summary["QLIKE"].idxmin()
    best_res   = results.get(best_model)
    if best_res is not None:
        cond_std  = np.sqrt(best_res.cond_var)
        var_99    = compute_var(cond_std, alpha=0.01)
        kupiec    = kupiec_test(rets, var_99, alpha=0.01)
        logger.info("VaR backtest (%s): violations=%d (%.2f%%) p=%.4f",
                    best_model, kupiec["violations"], kupiec["rate"]*100, kupiec["p_value"])
    else:
        var_99 = np.zeros(len(rets))
        kupiec = {"violations": 0, "rate": 0, "expected_rate": 0.01,
                  "p_value": np.nan, "reject_h0": False}

    # ── Save ──────────────────────────────────────────────────────────
    save_all(rets, results, summary, wf_results,
             None, var_99, kupiec, args.ticker, OUT)
    summary.to_csv(OUT / "tables" / "forecast_comparison.csv")
    model_table.to_csv(OUT / "tables" / "model_selection.csv")
    pd.Series({k: v.log_likelihood for k, v in results.items()
               if v is not None}).to_csv(OUT / "tables" / "log_likelihoods.csv")

    logger.info("="*60)
    logger.info("  Complete.  Best model (QLIKE): %s", best_model)
    logger.info("="*60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GARCH Volatility Forecasting")
    p.add_argument("--ticker",     default="SPY")
    p.add_argument("--start",      default="2010-01-01")
    p.add_argument("--end",        default=None)
    p.add_argument("--train-size", type=int, default=252)
    p.add_argument(
        "--evaluation-step",
        type=int,
        default=1,
        help="Evaluate every Nth observation (larger values make quick demos faster)",
    )
    p.add_argument("--offline",    action="store_true")
    main(p.parse_args())
