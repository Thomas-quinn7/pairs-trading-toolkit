"""Offline tests for the out-of-sample pairs-trading engine.

None of these touch the network: the engine's data fetch is injected with
synthetic panels via the ``panel=`` hook, so the whole OOS pipeline is exercised
on data with known properties.

The load-bearing test is ``test_no_lookahead``: it proves the signal and
position at time t are a pure function of data up to t, which is exactly the
defect the quarantined v1 engine (legacy/pair_trader_v1_lookahead.py) had.

Run:  python -m pytest tests/test_pairs.py -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import Backtesting as B  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic data helpers                                                      #
# --------------------------------------------------------------------------- #
def make_cointegrated_panel(seed=0, n=600, beta=1.5, phi=0.9, noise_sd=0.5):
    """A cointegrated pair A, B: B is a random walk, A = beta*B + const + OU noise.

    A - beta*B is stationary (OU), so the pair is cointegrated. Returns a price
    panel indexed by business days with columns 'A' and 'B'.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    logret = rng.normal(0.0002, 0.01, n)
    B_px = 100.0 * np.exp(np.cumsum(logret))
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + rng.normal(0.0, noise_sd)
    A_px = beta * B_px + 5.0 + e
    return pd.DataFrame({"A": A_px, "B": B_px}, index=idx)


# --------------------------------------------------------------------------- #
# Statistical building blocks                                                 #
# --------------------------------------------------------------------------- #
def test_ols_hedge_ratio_recovers_beta():
    panel = make_cointegrated_panel(seed=1, beta=1.75)
    beta = B.ols_hedge_ratio(panel["A"], panel["B"])
    assert abs(beta - 1.75) < 0.05


def test_half_life_recovers_known_value():
    """AR(1) with slope phi has Delta-regression half-life -ln2/(phi-1)."""
    rng = np.random.default_rng(2)
    phi = 0.92
    n = 6000
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + rng.standard_normal()
    hl = B.half_life(pd.Series(z))
    target = -np.log(2.0) / (phi - 1.0)
    assert abs(hl - target) / target < 0.15


def test_half_life_infinite_for_random_walk():
    rng = np.random.default_rng(3)
    rw = pd.Series(np.cumsum(rng.standard_normal(4000)))
    assert B.half_life(rw) > 100  # no mean reversion -> very long / inf


def test_engle_granger_finds_cointegrated_pair():
    panel = make_cointegrated_panel(seed=4)
    cands = B.scan_pairs_in_sample(panel, corr_threshold=0.5, stat_sig=0.05)
    assert any({c.s1, c.s2} == {"A", "B"} for c in cands)


def test_engle_granger_rejects_independent_random_walks():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2019-01-01", periods=600)
    x = 100 + np.cumsum(rng.standard_normal(600))
    y = 100 + np.cumsum(rng.standard_normal(600))
    panel = pd.DataFrame({"X": x, "Y": y}, index=idx)
    cands = B.scan_pairs_in_sample(panel, corr_threshold=0.0, stat_sig=0.01)
    assert not cands  # two independent random walks are not cointegrated


# --------------------------------------------------------------------------- #
# The look-ahead guard - the core integrity test                              #
# --------------------------------------------------------------------------- #
def test_no_lookahead():
    """Signal/position at time t must not depend on prices after t."""
    panel = make_cointegrated_panel(seed=7, n=600)
    idx = panel.index
    split = idx[300]

    common = dict(split_date=split.date().isoformat(), lookback_years=1,
                  Graphs="N", save_plots=False)
    res1 = B.backtest_pair_one_year("A", "B", panel=panel, **common)

    # Corrupt every price strictly after a cutoff inside the OOS window.
    cutoff = idx[350]
    panel2 = panel.copy()
    future = panel2.index > cutoff
    panel2.loc[future, "A"] *= 1.5
    panel2.loc[future, "B"] *= 0.7
    res2 = B.backtest_pair_one_year("A", "B", panel=panel2, **common)

    up_to = res1.df.index <= cutoff
    assert up_to.sum() > 20  # we actually compared a meaningful stretch
    assert np.allclose(res1.df.loc[up_to, "z"], res2.df.loc[up_to, "z"], atol=1e-9)
    assert (res1.df.loc[up_to, "position"].values == res2.df.loc[up_to, "position"].values).all()


def test_backtest_runs_and_reports_benchmark():
    panel = make_cointegrated_panel(seed=8, n=600)
    split = panel.index[300]
    res = B.backtest_pair_one_year("A", "B", panel=panel,
                                   split_date=split.date().isoformat(),
                                   lookback_years=1, Graphs="N", save_plots=False)
    assert np.isfinite(res.half_life)
    assert np.isfinite(res.bench_total_return)
    assert np.isfinite(res.bench_sharpe)
    assert len(res.df) > 0
