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
    # The OOS window is long enough for the paired bootstrap to run.
    assert np.isfinite(res.p_beat_bench)
    assert 0.0 <= res.p_beat_bench <= 1.0
    assert res.boot_return_p5 <= res.boot_return_p95
    assert res.boot_sharpe_p5 <= res.boot_sharpe_p95


# --------------------------------------------------------------------------- #
# Paired block bootstrap                                                      #
# --------------------------------------------------------------------------- #
def test_bootstrap_dominant_strategy_beats_benchmark():
    rng = np.random.default_rng(10)
    bench = rng.normal(0.0, 0.01, 252)
    strat = bench + 0.002  # same path shape, strictly higher mean every day
    boot = B.paired_block_bootstrap(strat, bench, n_sims=500, seed=1)
    assert boot["p_beat_bench"] > 0.99
    assert boot["return_p5"] <= boot["return_p50"] <= boot["return_p95"]


def test_bootstrap_symmetric_series_near_coin_flip():
    rng = np.random.default_rng(11)
    a = rng.normal(0.0005, 0.01, 500)
    b = rng.normal(0.0005, 0.01, 500)
    boot = B.paired_block_bootstrap(a, b, n_sims=500, seed=2)
    assert 0.2 < boot["p_beat_bench"] < 0.8


def test_bootstrap_deterministic_and_short_series_none():
    rng = np.random.default_rng(12)
    a = rng.normal(0.0, 0.01, 100)
    b = rng.normal(0.0, 0.01, 100)
    r1 = B.paired_block_bootstrap(a, b, n_sims=200, seed=3)
    r2 = B.paired_block_bootstrap(a, b, n_sims=200, seed=3)
    assert r1 == r2
    assert B.paired_block_bootstrap(a[:20], b[:20]) is None


# --------------------------------------------------------------------------- #
# Borrow / financing carry                                                    #
# --------------------------------------------------------------------------- #
def test_carry_costs_reduce_returns():
    """Holding costs must make an actively-trading run strictly worse."""
    panel = make_cointegrated_panel(seed=13, n=600)
    split = panel.index[300]
    common = dict(split_date=split.date().isoformat(), lookback_years=1,
                  Graphs="N", save_plots=False, ignore_gate=True, entry_z=0.5)
    free = B.backtest_pair_one_year("A", "B", panel=panel,
                                    borrow_rate=0.0, financing_rate=0.0, **common)
    costly = B.backtest_pair_one_year("A", "B", panel=panel,
                                      borrow_rate=0.10, financing_rate=0.25, **common)
    assert free.trades > 0  # the run actually held positions
    assert costly.total_return < free.total_return


def test_carry_costs_zero_when_flat():
    """A pair that never trades pays no carry: results identical at any rate."""
    rng = np.random.default_rng(14)
    idx = pd.bdate_range("2019-01-01", periods=600)
    # Independent random walks fail the cointegration gate -> position stays 0.
    panel = pd.DataFrame({"A": 100 + np.cumsum(rng.standard_normal(600)),
                          "B": 100 + np.cumsum(rng.standard_normal(600))}, index=idx)
    split = idx[300]
    common = dict(split_date=split.date().isoformat(), lookback_years=1,
                  Graphs="N", save_plots=False)
    free = B.backtest_pair_one_year("A", "B", panel=panel,
                                    borrow_rate=0.0, financing_rate=0.0, **common)
    costly = B.backtest_pair_one_year("A", "B", panel=panel,
                                      borrow_rate=0.10, financing_rate=0.25, **common)
    assert (free.df["position"] == 0).all()
    assert costly.total_return == pytest.approx(free.total_return)
