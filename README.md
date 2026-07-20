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

Transaction costs are charged on every unit change in position.

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

`python main.py` drives the out-of-sample engine in `Backtesting.py`: it scans
the universe for in-sample cointegrated pairs, then backtests each one on the
forward (out-of-sample) window and prints total/annualised return, Sharpe, max
drawdown, and trade count per pair. Plots are written to `charts/`.

Useful flags:

```bash
python main.py --tickers AAPL MSFT            # override the universe
python main.py --split-date 2024-01-01        # set the train/test boundary
python main.py --no-graphs                    # skip plotting
```

Edit `tickers.txt` to change the default search universe.

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
