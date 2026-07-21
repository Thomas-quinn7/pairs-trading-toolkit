"""Portfolio layer on top of the out-of-sample pairs engine.

Combines the OOS daily return streams of each traded pair into a single
portfolio equity curve. Two weighting schemes are provided:

- ``equal``       — 1/N across pairs, constant.
- ``inverse_vol`` — risk-balanced: weight_i proportional to 1/vol_i, where
  vol_i is the *trailing* rolling standard deviation of pair i's strategy
  returns, **lagged one bar**. Weights at time t are therefore a pure function
  of returns up to t-1 — causal by construction.

Why not the legacy Markowitz max-Sharpe optimiser? Optimising weights on the
same out-of-sample window you then report is look-ahead at the *portfolio*
level — the very class of bug the v1 engine had at the signal level. A
max-Sharpe weight vector fitted on the OOS returns "knows" which pairs did
well over the window it is being scored on. Until there is a walk-forward
weight-estimation scheme (fit on trailing window, apply forward), the honest
choices are equal weight or causal inverse-vol, so those are what run here.

Offline-testable: ``combine_pair_returns`` and ``perf_stats`` are pure
functions of an injected return panel; nothing here touches the network.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def combine_pair_returns(
    rets: pd.DataFrame,
    method: str = "inverse_vol",
    vol_window: int = 63,
    min_periods: int = 21,
    vol_floor: float = 1e-4,
) -> pd.DataFrame:
    """Combine per-pair daily strategy returns into portfolio returns.

    Parameters
    ----------
    rets : DataFrame of daily strategy returns, one column per pair. NaNs are
        treated as "not trading" (0 return).
    method : "equal" or "inverse_vol".
    vol_window / min_periods : trailing window for the inverse-vol estimate.
    vol_floor : lower clip on the vol estimate so a pair that sat flat (e.g.
        gated inactive by the cointegration re-test) cannot attract an
        exploding 1/vol weight.

    Returns a DataFrame with one weight column per pair (``w_<pair>``) and a
    ``port_ret`` column. Weights at time t use only returns up to t-1; during
    the warm-up (before ``min_periods`` of history) they fall back to 1/N.
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
              for m in ("equal", "inverse_vol")}
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

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            for col in rets.columns:
                ax1.plot(rets.index, (1.0 + rets[col].fillna(0)).cumprod(),
                         alpha=0.5, linewidth=1, label=col)
            for m, c in combos.items():
                ax1.plot(c.index, (1.0 + c["port_ret"]).cumprod(),
                         linewidth=2.2, label=f"portfolio ({m})")
            ax1.axhline(1.0, color="black", linestyle="--", alpha=0.4)
            ax1.set_title("OOS equity: individual pairs vs combined portfolio")
            ax1.set_ylabel("Equity (normalised)")
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=8)

            iv = combos["inverse_vol"]
            wcols = [c for c in iv.columns if c.startswith("w_")]
            ax2.stackplot(iv.index, [iv[c].values for c in wcols],
                          labels=[c[2:] for c in wcols], alpha=0.7)
            ax2.set_title("Inverse-vol weights (causal: trailing vol, lagged one bar)")
            ax2.set_ylabel("Weight")
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8, loc="upper right")

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
