# GARCH Volatility Forecasting

A from-scratch volatility-forecasting project covering GARCH(1,1), GJR-GARCH, EGARCH, HAR-RV, walk-forward QLIKE evaluation, and one-percent VaR calibration.

## Verified result

The committed [`results/`](results/README.md) evidence uses a deterministic synthetic return series and an evaluation stride of 21 observations:

- GARCH(1,1) QLIKE: -5.6585 (rank 1)
- GJR-GARCH QLIKE: -5.6559 (rank 2)
- HAR-RV QLIKE: -4.4588 (rank 3)
- One-percent VaR violations: 18 of 1,565 (1.15%)
- Kupiec p-value: 0.5598

Lower QLIKE is better. These are software and calibration results on generated data, not a claim about a traded asset.

## Reproduce

```bash
python -m pip install -r requirements.txt
python run_garch.py --offline --start 2018-01-01 --end 2024-01-01 --evaluation-step 21
python -m pytest -q
```

Run without `--offline` to download market data through the configured public-data source. Use `--evaluation-step 1` for full daily walk-forward evaluation; it is materially slower.

## Components

- Gaussian maximum-likelihood fitting for GARCH-family models
- Numerical safeguards for EGARCH log variance
- Realized-volatility HAR model
- Walk-forward MSE, MAE, RMSE, and QLIKE comparisons
- Kupiec proportion-of-failures VaR test
- Six diagnostic and result figures

The full pipeline writes to `outputs/`; compact reviewed artifacts are committed under [`results/`](results/README.md).

Python 3.10+ is recommended. This project is educational research, not investment advice.
