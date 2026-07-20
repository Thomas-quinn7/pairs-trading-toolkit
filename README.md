# Pairs Trading Toolkit

A statistical-arbitrage toolkit that screens a universe of stocks for
cointegrated pairs and backtests a mean-reversion spread strategy **out of
sample** — trained on a lookback window and evaluated on data it never saw
during calibration.

## Method

The engine (`Backtesting.py`) implements a proper Engle-Granger workflow:

- **Hedge ratio** — OLS slope `beta` from regressing one leg on the other, so
  the strategy trades a beta-hedged spread `s1 - beta*s2`, not a raw price ratio.
- **Cointegration test** — Augmented Dickey-Fuller (ADF) on the spread residual;
  a pair is only traded if the residual is stationary at the chosen significance.
- **Train/test split** — the residual mean/std used to form the z-score are
  estimated **only on the training window** (ending at the split date). The
  out-of-sample window runs forward from the split (~252 business days).
- **One-bar execution lag** — a signal formed on day *t* is executed on day
  *t+1*, so no bar trades on information from its own close.
- **Quarterly recalibration + OOS re-validation** — each calendar quarter the
  hedge ratio and residual stats are re-estimated on the trailing window and the
  ADF test is re-run; if the pair fails cointegration out of sample, trading is
  disabled for that period.
- **Half-life of mean reversion** — the Ornstein-Uhlenbeck half-life of the
  training spread is estimated (`half_life()`) and reported per pair, to justify
  the lookback and the expected holding horizon: a lookback far shorter than the
  half-life cannot see the reversion, and positions should be held on that order.
- **Passive benchmark** — each pair's OOS result is shown against an equal-weight
  buy-and-hold of the two legs (total return and Sharpe). The strategy is
  market-neutral, so the honest question is whether it beat simply holding the
  names on a risk-adjusted basis, net of costs.

Transaction costs are charged on every unit change in position. The default
universe is hand-picked and so carries **survivorship bias** — it is a
demonstration universe, not a bias-free backtest of a live selection rule.

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

`python main.py` drives the out-of-sample engine in `Backtesting.py`: it scans
the universe for in-sample cointegrated pairs, then backtests each one on the
forward (out-of-sample) window and prints per pair the half-life, total/annualised
return, Sharpe (each against the passive benchmark), max drawdown, and trade
count. Plots are written to `charts/`.

Useful flags:

```bash
python main.py --tickers AAPL MSFT            # override the universe
python main.py --split-date 2024-01-01        # set the train/test boundary
python main.py --no-graphs                    # skip plotting
```

Edit `tickers.txt` to change the default search universe.

The engine is unit-tested offline (synthetic panels injected via `panel=`, no
network) and the look-ahead demonstration below is reproducible:

```bash
python -m pytest tests/test_pairs.py -q       # incl. the look-ahead guard test
python demo_lookahead.py                       # quantifies the v1 bias
```

## Methodology note: how a look-ahead bug was caught

An earlier version of this repo (v1) normalised the z-score using the
**full-sample** mean and standard deviation of the spread:
`z = (ratio - ratio.mean()) / ratio.std()` over the *entire* history. That means
every historical signal implicitly "knew" the future distribution of the spread —
classic **look-ahead bias**. The symptom was an in-sample Sharpe that looked far
too good to be real; backtests that leak the future almost always do.

The fix was to rebuild the engine out of sample: estimate `mu`/`sd` (and the
hedge ratio) on a training window only, evaluate on a forward window the model
never saw, add a one-bar execution lag, and re-validate cointegration each
quarter so a pair that decoheres stops trading. That rebuilt engine is what
`Backtesting.py` / `python main.py` now runs.

**How big was the bias?** `demo_lookahead.py` runs the *same* mean-reversion rule
on synthetic spreads two ways — full-sample z (v1) vs causal trailing z (the
current engine) — and measures the Sharpe each reports:

```
data / persistence  full-sample   trailing        inflation
3.0y  phi=0.95            +2.20      +1.91       +0.29 (1.2x)
2.0y  phi=0.97            +1.85      +1.40       +0.45 (1.3x)
1.0y  phi=0.98            +1.82      +1.00       +0.81 (1.8x)
0.6y  phi=0.98            +2.12      +0.90       +1.22 (2.4x)
```

The look-ahead Sharpe is unearnable in real time, and the gap widens as history
shrinks and the spread gets more persistent — exactly the regime real pairs live
in, where the inflation is 1.8x–2.4x. The lesson is not that the number was huge
but that it was **invisible without the guard**: `tests/test_pairs.py::test_no_lookahead`
now asserts the signal at time *t* is unchanged when prices after *t* are
corrupted, so the bug cannot silently return.

The flawed v1 is kept, quarantined and clearly labelled, at
`legacy/pair_trader_v1_lookahead.py` — solely as the "before" side of this
write-up. Nothing in the active pipeline imports it.

## Portfolio optimisation (roadmap)

A Markowitz max-Sharpe portfolio optimiser (SLSQP, efficient frontier, live
risk-free rate from the 13-week T-bill `^IRX`) exists in the codebase but
currently lives in the legacy module because it consumed the flawed strategy's
returns. Reconnecting it on top of the out-of-sample engine is planned work; it
is not part of the default `python main.py` run today.

## Note

Research and learning code — not investment advice. Prices are pulled live via
yfinance.
