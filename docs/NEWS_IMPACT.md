# News-impact exhibit: isolate the mechanism

This exhibit develops the original project's `fig2_news_impact` comparison in `garch_volatility.zip`. It is a **parameterized illustration**, not a fitted model, a backtest, or a statement about current market risk.

Both models start from 20% annualized volatility (252 sessions). Their symmetric-innovation persistence is 0.95. GARCH uses alpha 0.10 and beta 0.85; GJR uses alpha 0.05, gamma 0.10 and beta 0.85. Both use omega = 0.05 × prior daily variance.

- GARCH: next variance = omega + alpha × shock² + beta × prior variance.
- GJR: next variance = omega + (alpha + gamma × I[shock < 0]) × shock² + beta × prior variance.

GARCH responds equally to equal positive and negative shocks. GJR assigns additional variance to a negative shock when gamma is positive. The comparison holds prior variance and the parameter set fixed; changing the shock does not refit a model.

The dashed 20% line is the prior annualized volatility, not a risk limit. Values shown are annualized one-session conditional volatility, not a forecast of annual losses. Volatility is not Value-at-Risk, default probability, or investment return.

![Illustrative news-impact curves](exhibit/news-impact.png)

Rebuild with `python scripts/build_news_impact_exhibit.py` in an environment containing Matplotlib. The script contains four independent hand-arithmetic checks. Exact 41-scenario values and parameter definitions are in [news-impact.json](exhibit/news-impact.json).

## Boundary of validation

This bounded exhibit verifies the two recursions at specified parameters. It does **not** validate the legacy full-pipeline forecast rankings or establish superiority of either model. The original archive is preserved. A separate audit is required for estimation convergence, next-step timing, common forecast targets, held-out comparisons and VaR coverage before relying on those broader outputs.
