# Verified results

The committed evidence was generated with:

```bash
python run_garch.py --offline --start 2018-01-01 --end 2024-01-01 --evaluation-step 21
```

This deterministic synthetic series was evaluated approximately monthly to keep the public smoke run fast while retaining walk-forward behavior.

Recorded forecast results:

- GARCH(1,1) QLIKE: -5.6585 (rank 1; 63 evaluations)
- GJR-GARCH QLIKE: -5.6559 (rank 2; 63 evaluations)
- HAR-RV QLIKE: -4.4588 (rank 3; 62 evaluations)
- One-percent VaR violations: 18 of 1,565 (1.15%)
- Kupiec proportion-of-failures p-value: 0.5598

Artifacts:

- `synthetic_forecast_comparison.csv` - walk-forward loss metrics and ranks
- `synthetic_model_selection.csv` - in-sample information criteria
- `synthetic_model_comparison.png` - forecast comparison
- `synthetic_var_backtest.png` - VaR exceptions and calibration annotation

These are synthetic execution and calibration results. They are not a claim about future volatility-forecast accuracy on a traded asset.
