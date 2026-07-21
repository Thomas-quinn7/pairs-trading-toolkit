"""Offline tests for the portfolio layer (portfolio.py).

Pure-function tests on synthetic return panels — no network, no engine run.
The load-bearing test is ``test_weights_are_causal``: inverse-vol weights at
time t must be a pure function of returns up to t-1, mirroring the signal-level
look-ahead guard in test_pairs.py.

Run:  python -m pytest tests/test_portfolio.py -q
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio import (  # noqa: E402
    combine_pair_returns,
    max_sharpe_weights,
    pair_returns_from_results,
    perf_stats,
    walk_forward_max_sharpe,
)


def make_returns_panel(seed=0, n=300, k=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    data = {f"P{i}": rng.normal(0.0003, 0.005 * (i + 1), n) for i in range(k)}
    return pd.DataFrame(data, index=idx)


def test_weights_sum_to_one_and_port_ret_is_weighted_sum():
    rets = make_returns_panel(seed=1)
    for method in ("equal", "inverse_vol", "max_sharpe"):
        out = combine_pair_returns(rets, method=method)
        wcols = [c for c in out.columns if c.startswith("w_")]
        w = out[wcols]
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-9)
        recomputed = (w.values * rets.values).sum(axis=1)
        assert np.allclose(out["port_ret"].values, recomputed, atol=1e-12)


def test_weights_are_causal():
    """Corrupting future returns must not change weights up to the cutoff."""
    rets = make_returns_panel(seed=2)
    cutoff = rets.index[200]

    out1 = combine_pair_returns(rets, method="inverse_vol")
    rets2 = rets.copy()
    rets2.loc[rets2.index > cutoff] *= 25.0  # violent future shock
    out2 = combine_pair_returns(rets2, method="inverse_vol")

    wcols = [c for c in out1.columns if c.startswith("w_")]
    up_to = out1.index <= cutoff
    assert up_to.sum() > 50
    assert np.allclose(out1.loc[up_to, wcols], out2.loc[up_to, wcols], atol=1e-12)


def test_inverse_vol_downweights_the_noisy_stream():
    """With equal means, the higher-vol stream should get the lower weight."""
    rets = make_returns_panel(seed=3, k=2)  # P1 has 2x the vol of P0
    out = combine_pair_returns(rets, method="inverse_vol")
    post_warmup = out.iloc[100:]
    assert (post_warmup["w_P0"] > post_warmup["w_P1"]).mean() > 0.95


def test_vol_floor_caps_flat_stream_weight():
    """A pair gated flat (all-zero returns) must not attract exploding weight."""
    rets = make_returns_panel(seed=4, k=2)
    rets["P1"] = 0.0
    out = combine_pair_returns(rets, method="inverse_vol")
    assert np.isfinite(out["port_ret"]).all()
    assert (out[["w_P0", "w_P1"]].values <= 1.0 + 1e-9).all()


def test_max_sharpe_weights_are_causal():
    """Corrupting future returns must not change walk-forward weights up to
    the cutoff — the Markowitz analogue of the signal-level look-ahead guard."""
    rets = make_returns_panel(seed=5)
    cutoff = rets.index[200]

    w1 = walk_forward_max_sharpe(rets)
    rets2 = rets.copy()
    rets2.loc[rets2.index > cutoff] *= 25.0
    w2 = walk_forward_max_sharpe(rets2)

    up_to = w1.index <= cutoff
    assert up_to.sum() > 50
    assert np.allclose(w1.loc[up_to], w2.loc[up_to], atol=1e-12)


def test_max_sharpe_prefers_the_better_stream():
    """A stream with real drift and low vol should out-weight a pure-noise one."""
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2023-01-02", periods=400)
    rets = pd.DataFrame({
        "good": rng.normal(0.0010, 0.004, 400),
        "noise": rng.normal(0.0000, 0.010, 400),
    }, index=idx)
    w = walk_forward_max_sharpe(rets, min_history=63)
    post_warmup = w.iloc[100:]
    assert post_warmup["good"].mean() > 0.7


def test_max_sharpe_degenerate_inputs_fall_back_cleanly():
    """Flat and collinear streams must yield finite, valid weights, not a crash."""
    idx = pd.bdate_range("2023-01-02", periods=200)
    flat = pd.DataFrame({"A": 0.0, "B": 0.0}, index=idx)
    w = walk_forward_max_sharpe(flat)
    assert np.isfinite(w.values).all()
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-9)

    rng = np.random.default_rng(7)
    base = rng.normal(0.0004, 0.006, 200)
    collinear = pd.DataFrame({"A": base, "B": base}, index=idx)  # singular cov
    w2 = walk_forward_max_sharpe(collinear)
    assert np.isfinite(w2.values).all()
    assert np.allclose(w2.sum(axis=1), 1.0, atol=1e-9)
    assert (w2.values >= -1e-9).all() and (w2.values <= 1.0 + 1e-9).all()


def test_max_sharpe_single_fit_is_sane():
    """Direct check of the one-window optimiser: bounds respected, sums to 1."""
    rets = make_returns_panel(seed=8, n=150)
    w = max_sharpe_weights(rets)
    assert w.shape == (3,)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all() and (w <= 1.0 + 1e-9).all()


def test_perf_stats_known_values():
    ret = pd.Series([0.01] * 252)
    s = perf_stats(ret)
    assert abs(s["total_return"] - (1.01 ** 252 - 1.0)) < 1e-9
    assert s["max_dd"] == 0.0
    assert s["sharpe"] == 0.0  # zero variance -> Sharpe reported as 0


def test_pair_returns_from_results_roundtrip():
    """End-to-end with the real engine on a synthetic panel: the extracted
    return column must reproduce the engine's own equity curve."""
    import Backtesting as B
    from tests.test_pairs import make_cointegrated_panel

    panel = make_cointegrated_panel(seed=11, n=600)
    split = panel.index[300]
    res = B.backtest_pair_one_year("A", "B", panel=panel,
                                   split_date=split.date().isoformat(),
                                   lookback_years=1, Graphs="N", save_plots=False)

    class _PC:  # minimal stand-in for PairCandidate
        pass

    rets = pair_returns_from_results([(_PC(), res)])
    assert rets.shape[1] == 1
    eq_from_rets = (1.0 + rets.iloc[:, 0]).cumprod()
    assert np.allclose(eq_from_rets.values, res.df["equity"].values, atol=1e-12)
