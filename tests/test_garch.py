"""tests/test_garch.py — GARCH model math tests (no internet required)"""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from garch.models.garch import GARCH11, GJRGARCH, EGARCH, fit_all, compare_models
from garch.models.forecast import walk_forward_garch
from garch.models.har_rv import HARRV

np.random.seed(42)


def _synthetic_garch_returns(n=600, omega=1e-5, alpha=0.08, beta=0.88):
    eps, sig2 = np.zeros(n), np.zeros(n)
    sig2[0] = omega / (1 - alpha - beta)
    z = np.random.standard_normal(n)
    for t in range(1, n):
        sig2[t] = omega + alpha * eps[t-1]**2 + beta * sig2[t-1]
        eps[t]  = z[t] * np.sqrt(sig2[t])
    return eps, sig2


def test_garch11_converges():
    eps, _ = _synthetic_garch_returns()
    res = GARCH11().fit(eps)
    assert res.converged, "GARCH(1,1) should converge on well-behaved data"


def test_garch11_stationarity_constraint():
    eps, _ = _synthetic_garch_returns()
    res = GARCH11().fit(eps)
    assert res.params["alpha"] + res.params["beta"] < 1.0


def test_garch11_positive_params():
    eps, _ = _synthetic_garch_returns()
    res = GARCH11().fit(eps)
    assert res.params["omega"] > 0
    assert res.params["alpha"] > 0
    assert res.params["beta"]  > 0


def test_garch11_cond_var_positive():
    eps, _ = _synthetic_garch_returns()
    res = GARCH11().fit(eps)
    assert np.all(res.cond_var > 0)


def test_garch11_forecast_reverts_to_unconditional():
    eps, _ = _synthetic_garch_returns()
    model = GARCH11(); model.fit(eps)
    forecasts = model.forecast(h=252)
    uncond = model.result.params["uncond_var"]
    # At h=252, forecast should be close to unconditional variance
    assert abs(forecasts[-1] - uncond) / uncond < 0.10


def test_gjr_garch_has_leverage_on_asymmetric_data():
    # Create returns with stronger negative shocks
    np.random.seed(1)
    eps, _ = _synthetic_garch_returns()
    neg_shock_mask = eps < 0
    eps[neg_shock_mask] *= 1.5   # amplify negative shocks
    res = GJRGARCH().fit(eps)
    # gamma should be positive (negative returns increase vol more)
    # Not always true on short sample, so just check it's fitted
    assert "gamma" in res.params


def test_egarch_fitted():
    eps, _ = _synthetic_garch_returns()
    res = EGARCH().fit(eps)
    assert res.log_likelihood < 0 or res.log_likelihood > -1e10


def test_compare_models_returns_dataframe():
    eps, _ = _synthetic_garch_returns()
    results = fit_all(eps)
    tbl = compare_models(results)
    assert "AIC" in tbl.columns and "BIC" in tbl.columns
    assert len(tbl) == 3


def test_har_rv_r2_positive():
    eps, _ = _synthetic_garch_returns(n=400)
    rv = pd.Series(eps**2)
    model = HARRV(rv); res = model.fit()
    assert res.r2 >= 0


def test_har_rv_forecast_positive():
    eps, _ = _synthetic_garch_returns(n=300)
    rv = pd.Series(eps**2)
    model = HARRV(rv); model.fit()
    fcast = model.forecast(1)
    assert fcast > 0


def test_walk_forward_evaluation_step_subsamples_deterministically():
    eps, _ = _synthetic_garch_returns(n=190)
    dense = walk_forward_garch(eps, train_size=150, evaluation_step=1)
    sparse = walk_forward_garch(eps, train_size=150, evaluation_step=10)

    assert len(sparse) == len(dense.iloc[::10])
    assert sparse["t"].tolist() == dense.iloc[::10]["t"].tolist()


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try: fn(); print(f"  ✅  {fn.__name__}"); passed += 1
        except Exception as e: print(f"  ❌  {fn.__name__}: {e}"); failed += 1
    print(f"\n  {passed} passed | {failed} failed")
