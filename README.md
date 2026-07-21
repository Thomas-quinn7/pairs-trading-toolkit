# Pairs Trading Toolkit

A statistical-arbitrage toolkit that screens a universe of stocks for
cointegrated pairs and backtests a mean-reversion spread strategy **out of
sample** — trained on a lookback window and evaluated on data it never saw
during calibration.

## Method

The engine (`Backtesting.py`) implements a proper Engle-Granger workflow:

- **Hedge ratio** — OLS slope `beta` from regressing one leg on the other, so
  the strategy trades a beta-hedged spread `s1 - beta*s2`, not a raw price ratio.
- **Cointegration test** — the Engle-Granger test (`statsmodels.coint`) on the
  pair, not a plain ADF on the residual: because `beta` is *estimated* by OLS,
  standard Dickey-Fuller critical values are too lenient (the fit makes the
  residual look more stationary than it is); Engle-Granger/MacKinnon critical
  values correct for that. The screen tests only the spread — the object the
  strategy actually trades and re-validates — and the same test gates the
  quarterly re-check. With N tickers the screen runs N(N−1)/2 tests at one
  significance level with no multiple-testing correction, so some false pairs
  will slip through by chance; the OOS re-validation is the backstop.
- **Train/test split** — the residual mean/std used to form the z-score are
  estimated **only on the training window** (ending at the split date). The
  out-of-sample window runs forward from the split (~252 business days).
- **One-bar execution lag** — a signal formed on day *t* is executed on day
  *t+1*, so no bar trades on information from its own close.
- **Quarterly recalibration + OOS re-validation** — each calendar quarter the
  hedge ratio and residual stats are re-estimated on the trailing window and the
  Engle-Granger test is re-run; if the pair fails cointegration out of sample,
  trading is disabled for that period.
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
python main.py --portfolio                    # combine pairs into a portfolio (see below)
python main.py --corr-threshold 0.8 --stat-sig 0.05   # relax the in-sample screen
```

If nothing passes the screen at the (deliberately strict) defaults, the run
reports the closest candidates and their p-values and exits cleanly — an
honest "no trade" result, not an error.

Edit `tickers.txt` to change the default search universe.

The engine and portfolio layer are unit-tested offline (synthetic panels
injected via `panel=`, no network) and the look-ahead demonstration below is
reproducible:

```bash
python -m pytest tests -q                      # incl. both look-ahead guard tests
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

## Portfolio layer (`--portfolio`)

`python main.py --portfolio` combines the OOS daily return streams of every
traded pair into one portfolio (`portfolio.py`), reporting total/annualised
return, vol, Sharpe, and max drawdown for two weighting schemes, plus the
average pairwise correlation of the pair return streams (the diversification
evidence). A combined equity/weights chart is written to
`charts/pairs_portfolio_oos.png`.

- **equal** — 1/N, constant.
- **inverse_vol** — weight ∝ 1/vol, where vol is the *trailing* rolling
  standard deviation of each pair's strategy returns, **lagged one bar** — the
  weight applied over bar *t* is known at the close of *t−1*, so the scheme is
  causal by construction. `tests/test_portfolio.py::test_weights_are_causal`
  guards this the same way `test_no_lookahead` guards the signal.

**Why not the legacy Markowitz max-Sharpe optimiser?** Optimising weights on
the same out-of-sample window you then report is look-ahead at the *portfolio*
level — a max-Sharpe weight vector fitted on the OOS returns "knows" which
pairs did well over the window it is scored on. The legacy optimiser (SLSQP,
efficient frontier, live `^IRX` risk-free rate) stays quarantined until it can
be driven walk-forward: fit weights on a trailing window, apply them forward.
That remains the roadmap item.

### A worked example of why the strict screen matters

On the current universe (July 2025 split), the strict default screen
(Engle-Granger p<0.01) admits **nothing** — the closest pair sits at p≈0.016.
Relaxing to `--stat-sig 0.05` admits three pairs, and out of sample two of
them lose 43% and 68% respectively. The pairs that "almost" pass the test are
exactly the ones the test exists to keep you out of; a no-trade year is a
result, not a failure.

## Note

Research and learning code — not investment advice. Prices are pulled live via
yfinance.
