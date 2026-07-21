"""Portfolio layer on top of the out-of-sample pairs engine.

Combines the OOS daily return streams of each traded pair into a single
portfolio equity curve. Three weighting schemes are provided:

- ``equal``       — 1/N across pairs, constant.
- ``inverse_vol`` — risk-balanced: weight_i proportional to 1/vol_i, where
  vol_i is the *trailing* rolling standard deviation of pair i's strategy
  returns, **lagged one bar**. Weights at time t are therefore a pure function
  of returns up to t-1 — causal by construction.
- ``max_sharpe``  — walk-forward Markowitz: every ``rebalance`` bars, long-only
  max-Sharpe weights are fitted by SLSQP on the trailing ``fit_window`` of
  returns **up to the previous bar** and held until the next refit. This is
  the honest reconnection of the legacy optimiser: fitting max-Sharpe weights
  on the same OOS window you then report would be look-ahead at the
  *portfolio* level (the weight vector would "know" which pairs did well over
  the window it is scored on) — the very class of bug the v1 engine had at
  the signal level. Walk-forward fitting removes that: at every bar the
  weights are a pure function of past returns, guarded by
  ``tests/test_portfolio.py::test_max_sharpe_weights_are_causal``.

The risk-free rate is taken as zero: the pair strategies are self-financing
long/short books, so their raw daily returns are already excess-return-like.
(The legacy module's live ^IRX fetch belonged to its buy-and-hold display,
not to the weight fit.)

Offline-testable: ``combine_pair_returns`` and ``perf_stats`` are pure
functions of an injected return panel; nothing here touches the network.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def max_sharpe_weights(window: pd.DataFrame, ridge: float = 1e-4) -> np.ndarray:
    """Long-only max-Sharpe weights for one fit window of daily returns (rf=0).

    SLSQP on -Sharpe with weights in [0, 1] summing to 1. The covariance gets
    a small ridge (scaled to its average variance) so a near-flat or collinear
    stream cannot make the problem singular. Any optimiser failure falls back
    to equal weight — a portfolio must always have valid weights.
    """
    from scipy.optimize import minimize

    mu = window.mean().values
    cov = window.cov().values
    n = len(mu)
    eq = np.full(n, 1.0 / n)
    if n == 1:
        return eq
    scale = max(float(np.trace(cov)) / n, 1e-12)
    cov = cov + ridge * scale * np.eye(n)

    def neg_sharpe(w: np.ndarray) -> float:
        vol = float(np.sqrt(w @ cov @ w))
        return -(float(w @ mu) / vol) if vol > 0 else 0.0

    try:
        res = minimize(
            neg_sharpe, eq, method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}],
            options={"maxiter": 200},
        )
        w = res.x if res.success and np.isfinite(res.x).all() else eq
    except Exception:
        w = eq
    w = np.clip(w, 0.0, 1.0)
    s = float(w.sum())
    return w / s if s > 0 else eq


def walk_forward_max_sharpe(
    rets: pd.DataFrame,
    fit_window: int = 126,
    rebalance: int = 21,
    min_history: int = 63,
) -> pd.DataFrame:
    """Walk-forward max-Sharpe weights, one row per bar.

    Until ``min_history`` bars of returns exist, weights are 1/N. From then on,
    every ``rebalance`` bars the weights are refitted on the trailing
    ``fit_window`` bars **ending at the previous bar** (``iloc[:i]`` — the bar
    being weighted is never in its own fit window) and held until the next
    refit. Causal by construction.
    """
    rets = rets.astype(float).fillna(0.0)
    n = rets.shape[1]
    w = np.full(n, 1.0 / n)
    rows = []
    for i in range(len(rets)):
        if i >= min_history and (i - min_history) % rebalance == 0:
            w = max_sharpe_weights(rets.iloc[max(0, i - fit_window):i])
        rows.append(w.copy())
    return pd.DataFrame(rows, index=rets.index, columns=rets.columns)


def combine_pair_returns(
    rets: pd.DataFrame,
    method: str = "inverse_vol",
    vol_window: int = 63,
    min_periods: int = 21,
    vol_floor: float = 1e-4,
    fit_window: int = 126,
    rebalance: int = 21,
) -> pd.DataFrame:
    """Combine per-pair daily strategy returns into portfolio returns.

    Parameters
    ----------
    rets : DataFrame of daily strategy returns, one column per pair. NaNs are
        treated as "not trading" (0 return).
    method : "equal", "inverse_vol", or "max_sharpe" (walk-forward Markowitz).
    vol_window / min_periods : trailing window for the inverse-vol estimate.
    vol_floor : lower clip on the vol estimate so a pair that sat flat (e.g.
        gated inactive by the cointegration re-test) cannot attract an
        exploding 1/vol weight.
    fit_window / rebalance : trailing fit length and refit cadence (bars) for
        the walk-forward max-Sharpe scheme.

    Returns a DataFrame with one weight column per pair (``w_<pair>``) and a
    ``port_ret`` column. Weights at time t use only returns up to t-1; during
    the warm-up (before enough history exists) they fall back to 1/N.
    """
    if rets.shape[1] == 0:
        raise ValueError("No return columns to combine.")
    rets = rets.astype(float).fillna(0.0)
    n = rets.shape[1]

    if method == "equal":
        w = pd.DataFrame(1.0 / n, index=rets.index, columns=rets.columns)
    elif method == "inverse_vol":
        vol = rets.rolling(vol_window, min_periods=min_periods).std(ddof=1)
        # shift(1): the weight applied over bar t is known at the close of t-1.
        inv = 1.0 / vol.clip(lower=vol_floor).shift(1)
        rowsum = inv.sum(axis=1)
        w = inv.div(rowsum.where(rowsum > 0), axis=0)
        w = w.fillna(1.0 / n)  # warm-up: no trailing history yet -> 1/N
    elif method == "max_sharpe":
        w = walk_forward_max_sharpe(rets, fit_window=fit_window, rebalance=rebalance)
    else:
        raise ValueError(f"Unknown weighting method: {method!r}")

    out = w.add_prefix("w_")
    out["port_ret"] = (w * rets).sum(axis=1)
    return out


def perf_stats(ret: pd.Series) -> Dict[str, float]:
    """Total/annualised return, annualised vol, Sharpe, max drawdown."""
    ret = pd.Series(ret).astype(float).dropna()
    if len(ret) == 0:
        return {k: float("nan") for k in
                ("total_return", "ann_return", "ann_vol", "sharpe", "max_dd")}
    eq = (1.0 + ret).cumprod()
    total = float(eq.iloc[-1] - 1.0)
    sd = float(ret.std(ddof=1)) if len(ret) > 1 else 0.0
    sharpe = float(ret.mean() / sd * np.sqrt(252.0)) if sd > 0 else 0.0
    ann = float((1.0 + total) ** (252.0 / len(ret)) - 1.0)
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "total_return": total,
        "ann_return": ann,
        "ann_vol": sd * float(np.sqrt(252.0)),
        "sharpe": sharpe,
        "max_dd": dd,
    }


def pair_returns_from_results(results: List[Tuple[object, object]]) -> pd.DataFrame:
    """Extract the per-pair OOS daily return panel from engine results.

    ``results`` is the list of (PairCandidate, BacktestResult) tuples returned
    by ``backtest_all_pairs_one_year``; each BacktestResult.df carries a
    ``ret`` column (net of transaction costs).
    """
    cols = {}
    for pc, res in results:
        cols[f"{res.s1}/{res.s2}"] = res.df["ret"]
    if not cols:
        raise ValueError("No backtest results to build a portfolio from.")
    return pd.DataFrame(cols).sort_index()


def summarize_portfolio(
    results: List[Tuple[object, object]],
    vol_window: int = 63,
    show_plots: bool = True,
    save_plots: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Print and (optionally) plot the combined OOS portfolio vs equal weight.

    Returns {"equal": stats, "inverse_vol": stats} for programmatic use.
    """
    rets = pair_returns_from_results(results)
    n_pairs = rets.shape[1]

    combos = {m: combine_pair_returns(rets, method=m, vol_window=vol_window)
              for m in ("equal", "inverse_vol", "max_sharpe")}
    stats = {m: perf_stats(c["port_ret"]) for m, c in combos.items()}

    # Diversification evidence: average pairwise correlation of the pair
    # strategy return streams (low correlation is where the portfolio benefit
    # comes from).
    avg_corr = float("nan")
    if n_pairs > 1:
        cm = rets.corr().values
        iu = np.triu_indices(n_pairs, k=1)
        avg_corr = float(np.nanmean(cm[iu]))

    print(f"\n=== OOS portfolio of {n_pairs} pair strategies "
          f"(avg pairwise corr {avg_corr:.2f}) ===")
    for m, s in stats.items():
        print(f"{m:>12s} | total={s['total_return']:.2%} | ann={s['ann_return']:.2%} | "
              f"vol={s['ann_vol']:.2%} | sharpe={s['sharpe']:.2f} | maxDD={s['max_dd']:.2%}")

    if show_plots or save_plots:
        try:
            import matplotlib
            if not show_plots:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            import plotstyle as ps
            ps.apply_style()

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Individual pairs are context, not the headline: muted, thin, one
            # legend entry — the three portfolio schemes carry the identity
            # colors (fixed slots, never cycled).
            for k, col in enumerate(rets.columns):
                ax1.plot(rets.index, (1.0 + rets[col].fillna(0)).cumprod(),
                         color=ps.MUTED, alpha=0.45, linewidth=1.0,
                         label=f"individual pairs ({n_pairs})" if k == 0 else None)
            scheme_slots = {"equal": 0, "inverse_vol": 1, "max_sharpe": 2}
            for m, c in combos.items():
                ax1.plot(c.index, (1.0 + c["port_ret"]).cumprod(),
                         color=ps.series_color(scheme_slots[m]),
                         linewidth=2.2, label=f"portfolio ({m})")
            ax1.axhline(1.0, color=ps.BASELINE, linestyle="--", linewidth=1.0)
            ax1.set_title("OOS equity: portfolio schemes vs individual pairs")
            ax1.set_ylabel("Equity (normalised)")
            ax1.legend(loc="best")

            # Weights panel shows the walk-forward max-Sharpe allocation (the
            # scheme with time-varying structure worth inspecting). Identity
            # per pair via the categorical slots; past 8 pairs, fold into
            # "Other" rather than cycling hues.
            ms = combos["max_sharpe"]
            wcols = [c for c in ms.columns if c.startswith("w_")]
            if len(wcols) > 8:
                keep = list(ms[wcols].mean().sort_values(ascending=False).index[:7])
                other = [c for c in wcols if c not in keep]
                stack = [ms[c].values for c in keep] + [ms[other].sum(axis=1).values]
                labels = [c[2:] for c in keep] + [f"Other ({len(other)})"]
                colors = [ps.series_color(k) for k in range(7)] + [ps.MUTED]
            else:
                stack = [ms[c].values for c in wcols]
                labels = [c[2:] for c in wcols]
                colors = [ps.series_color(k) for k in range(len(wcols))]
            ax2.stackplot(ms.index, stack, labels=labels, colors=colors,
                          alpha=0.85, edgecolor=ps.SURFACE, linewidth=1.0)
            ax2.set_title("Walk-forward max-Sharpe weights "
                          "(fit on trailing window, applied forward)")
            ax2.set_ylabel("Weight")
            ax2.set_ylim(0, 1)
            ax2.legend(loc="upper right")

            plt.tight_layout()
            if save_plots:
                charts_dir = os.path.join(os.path.dirname(__file__), "charts")
                os.makedirs(charts_dir, exist_ok=True)
                plt.savefig(os.path.join(charts_dir, "pairs_portfolio_oos.png"),
                            dpi=300, bbox_inches="tight")
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
        except Exception as exc:  # plotting must never kill the run
            print(f"[WARN] Portfolio plot failed: {exc}")

    return stats
