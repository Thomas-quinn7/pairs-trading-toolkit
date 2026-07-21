from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import os
from statsmodels.tsa.stattools import coint

# yfinance and matplotlib are imported lazily (inside fetch_prices and the
# plotting branch) so the engine can be imported and unit-tested on injected
# synthetic data with no network or GUI dependency.


def fetch_prices(
    tickers: Iterable[str] | str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch close prices for tickers between start and end (YYYY-MM-DD)."""
    import yfinance as yf

    if isinstance(tickers, str):
        tickers = [tickers]
    df = yf.download(list(tickers), start=start, end=end, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.astype(float).dropna(how="all")


def ols_hedge_ratio(s1: pd.Series, s2: pd.Series) -> float:
    """OLS slope (minimize (s1 - beta*s2)^2) -> beta = Cov/Var."""
    s1, s2 = s1.align(s2, join="inner")
    s1 = s1.astype(float)
    s2 = s2.astype(float)
    v = np.var(s2.values, ddof=1)
    if v == 0 or np.isnan(v):
        return 1.0
    c = np.cov(np.vstack([s1.values, s2.values]), ddof=1)[0, 1]
    beta = c / v
    if not np.isfinite(beta):
        beta = 1.0
    return float(beta)


def half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life of mean reversion, in bars (trading days).

    Fits the discrete OU relation  d(spread)_t = a + lambda * spread_{t-1} + e_t
    by OLS. For a mean-reverting series lambda < 0 and the half-life - the time
    to close half the gap to the mean - is -ln(2) / lambda. A non-reverting
    series (lambda >= 0) returns inf. The half-life justifies the trading
    lookback and expected holding horizon: entries should be held on the order
    of one half-life, and a lookback far shorter than it cannot see the
    reversion.
    """
    s = pd.Series(spread).dropna().astype(float)
    if len(s) < 10:
        return float("nan")
    lag = s.shift(1)
    delta = s - lag
    frame = pd.concat([delta, lag], axis=1).dropna()
    delta_v = frame.iloc[:, 0].values
    lag_v = frame.iloc[:, 1].values
    X = np.column_stack([np.ones(len(lag_v)), lag_v])
    try:
        lam = np.linalg.lstsq(X, delta_v, rcond=None)[0][1]
    except Exception:
        return float("nan")
    if lam >= 0 or not np.isfinite(lam):
        return float("inf")
    return float(-np.log(2.0) / lam)


@dataclass
class PairCandidate:
    s1: str
    s2: str
    beta: float
    pvalue: float
    method: str  # "spread" or "ratio"


def scan_pairs_in_sample(
    prices: pd.DataFrame,
    corr_threshold: float = 0.9,
    stat_sig: float = 0.01,
) -> List[PairCandidate]:
    """Scan all pairs in the price panel and return cointegration candidates.

    The spread (s1 - beta*s2) is tested with the Engle-Granger cointegration
    test (``statsmodels.coint``), NOT a plain ADF on the residual: because beta
    is *estimated* by OLS, standard Dickey-Fuller critical values are too
    lenient (the fit makes the residual look more stationary than it is), and
    Engle-Granger/MacKinnon critical values correct for that.

    Only the spread is tested, because the spread is what the engine actually
    trades and re-validates each quarter. (An earlier version also admitted
    pairs on an ADF of the price *ratio*; such a pair could pass the screen yet
    fail the spread test the engine gates on, and would then sit inactive for
    the whole OOS window — screening on a statistic you do not trade is
    incoherent.)

    Note the screen tests many pairs at one significance level with no
    multiple-testing correction: with N tickers there are N*(N-1)/2 tests, so
    at stat_sig=0.01 roughly 1 in 100 non-cointegrated pairs will still slip
    through by chance. The OOS re-validation is the backstop for that.
    """
    tickers = list(prices.columns)
    corr = prices.corr().abs()
    cands: List[PairCandidate] = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            if corr.loc[a, b] < corr_threshold:
                continue
            s1 = prices[a].dropna()
            s2 = prices[b].dropna()
            common = s1.index.intersection(s2.index)
            s1 = s1.loc[common]
            s2 = s2.loc[common]
            if len(s1) < 50:
                continue

            beta = ols_hedge_ratio(s1, s2)
            try:
                p_spread = float(coint(s1, s2, maxlag=1, autolag="aic")[1])
            except Exception:
                continue
            if p_spread < stat_sig:
                cands.append(PairCandidate(a, b, beta, p_spread, "spread"))
    # Sort by smallest p-value
    cands.sort(key=lambda x: x.pvalue)
    return cands


@dataclass
class BacktestResult:
    s1: str
    s2: str
    beta: float
    method: str
    entry: float
    exit: float
    total_return: float
    ann_return: float
    sharpe: float
    max_dd: float
    trades: int
    df: pd.DataFrame
    half_life: float = float("nan")        # OU half-life of the training spread (bars)
    bench_total_return: float = float("nan")  # equal-weight buy-hold of the pair, OOS
    bench_sharpe: float = float("nan")


def _zscore(x: pd.Series, mu: float, sd: float) -> pd.Series:
    sd = sd if sd > 0 else 1e-8
    return (x - mu) / sd


def backtest_pair_one_year(
    s1: str,
    s2: str,
    split_date: str,
    lookback_years: int = 2,
    entry_z: float = 1.0,
    exit_z: float = 0.2,
    stop_z: float = 4.5,
    tc: float = 0.001,
    end_date: Optional[str] = None,
    stat_sig: float = 0.01,
    Graphs: str = "Y",
    save_plots: bool = True,
    z_step: float = 0.5,
    max_units: int = 5,
    debug: bool = False,
    ignore_gate: bool = False,
    panel: Optional[pd.DataFrame] = None,
) -> BacktestResult:
    """Train on lookback ending at split_date, then test next ~252 BDays.

    Enhancements:
    - One-bar execution delay: signals today are executed next bar.
    - Quarterly re-test/recalibration: each calendar quarter, re-estimate beta and
      residual mean/std on the trailing lookback window and disable trading if
      the Engle-Granger cointegration test fails the threshold.
    """
    split = pd.to_datetime(split_date)
    train_start = (split - pd.DateOffset(years=lookback_years)).date().isoformat()
    test_end = pd.to_datetime(end_date) if end_date else (split + pd.tseries.offsets.BDay(252))

    if panel is None:
        panel = fetch_prices([s1, s2], start=train_start, end=test_end.date().isoformat())
    s1_all = panel[s1].dropna()
    s2_all = panel[s2].dropna()
    common = s1_all.index.intersection(s2_all.index)
    s1_all = s1_all.loc[common]
    s2_all = s2_all.loc[common]

    def _calibrate(train_end: pd.Timestamp) -> Tuple[float, float, float, bool, float]:
        start_dt = pd.to_datetime(train_start)
        window_start = max(start_dt, train_end - pd.DateOffset(years=lookback_years))
        s1_tr = s1_all.loc[window_start:train_end]
        s2_tr = s2_all.loc[window_start:train_end]
        beta_loc = ols_hedge_ratio(s1_tr, s2_tr)
        spread_tr = (s1_tr - beta_loc * s2_tr).dropna()
        mu_loc = float(spread_tr.mean())
        sd_loc = float(spread_tr.std(ddof=1))
        active = True
        p_val = float("nan")
        try:
            # Engle-Granger, not plain ADF: beta_loc is estimated, so DF
            # critical values on the residual would be too lenient.
            p_val = float(coint(s1_tr, s2_tr, maxlag=1, autolag="aic")[1])
            active = bool(p_val < stat_sig)
        except Exception:
            active = False
        return float(beta_loc), mu_loc, sd_loc, active, p_val

    beta, mu, sd, active, p_val = _calibrate(split)

    # Half-life of mean reversion on the initial training spread (justifies the
    # lookback / expected holding horizon). Uses only in-sample data.
    _win_start = max(pd.to_datetime(train_start), split - pd.DateOffset(years=lookback_years))
    _spread_train = (s1_all.loc[_win_start:split] - beta * s2_all.loc[_win_start:split]).dropna()
    hl = half_life(_spread_train)

    # Out-of-sample window (next ~252 BDays)
    test_idx = s1_all.index[s1_all.index > split]
    if len(test_idx) == 0:
        raise ValueError("No out-of-sample data after split_date.")
    test_idx = test_idx[:252]
    s1_test = s1_all.loc[test_idx]
    s2_test = s2_all.loc[test_idx]

    # Returns for hedged spread components
    r1 = s1_test.pct_change().fillna(0.0)
    r2 = s2_test.pct_change().fillna(0.0)

    pos = 0  # -1 short spread, +1 long spread
    eq = 1.0
    eq_curve = []
    rets = []
    trades = 0

    z_series: List[float] = []
    beta_series: List[float] = []
    signal_series: List[int] = []
    position_series: List[int] = []
    active_series: List[bool] = []
    pval_series: List[float] = []
    tc_carry = 0.0
    last_calib_quarter = split.to_period("Q")

    test_dates = list(test_idx)
    for i, ts in enumerate(test_dates):
        # Record current position for this bar (used for equity calc at this ts)
        position_series.append(int(pos))
        # Quarterly re-test and re-calibration at quarter changes
        current_q = ts.to_period("Q")
        if current_q != last_calib_quarter:
            train_end = test_dates[i - 1] if i > 0 else split
            beta, mu, sd, active, p_val = _calibrate(train_end)
            last_calib_quarter = current_q

        # Today's z-score using current calibration
        spread_t = float(s1_test.loc[ts] - beta * s2_test.loc[ts])
        sd_eff = sd if sd > 0 else 1e-8
        z_t = (spread_t - mu) / sd_eff
        z_series.append(float(z_t))
        beta_series.append(float(beta))
        # Respect gating unless ignore_gate overrides it for analysis
        is_active = True if ignore_gate else bool(active)
        active_series.append(is_active)
        pval_series.append(float(p_val) if p_val == p_val else np.nan)

        # PnL from yesterday's position (one-bar delay)
        gross_ret = (r1.loc[ts] - beta * r2.loc[ts]) if i > 0 else 0.0
        ret = pos * float(gross_ret) - tc_carry
        eq *= (1.0 + ret)
        eq_curve.append(eq)
        rets.append(ret)
        tc_carry = 0.0  # consumed

        # Decide desired position (multi-unit ladder) from today's z; applies next bar
        desired_pos = pos
        if not is_active:
            desired_pos = 0
        else:
            abs_z = abs(z_t)
            if abs_z <= exit_z or abs_z < entry_z:
                target_units = 0
            else:
                # 1 unit at entry threshold, then +1 per z_step beyond, capped
                target_units = int(min(max_units, 1 + np.floor((abs_z - entry_z) / max(z_step, 1e-8))))

            if z_t > entry_z:
                desired_pos = -target_units
            elif z_t < -entry_z:
                desired_pos = +target_units
            else:
                desired_pos = 0

            # Stop loss only closes existing positions; does not block new entries
            if abs(z_t) >= stop_z and pos != 0:
                desired_pos = 0

        # Compute unit change and costs to apply next bar
        delta_units = int(desired_pos - pos)
        sig_val = 0
        if delta_units != 0:
            trades += abs(delta_units)
            tc_carry = abs(delta_units) * 2 * tc
            sig_val = 1 if delta_units > 0 else -1
        # Position update applies next bar
        pos = desired_pos
        signal_series.append(sig_val)

    df = pd.DataFrame(
        {
            "s1": s1_test,
            "s2": s2_test,
            "z": pd.Series(z_series, index=test_idx, dtype=float),
            "beta": pd.Series(beta_series, index=test_idx, dtype=float),
            "equity": pd.Series(eq_curve, index=test_idx, dtype=float),
            "ret": pd.Series(rets, index=test_idx, dtype=float),
            "signal": pd.Series(signal_series, index=test_idx, dtype=int),
            "position": pd.Series(position_series, index=test_idx, dtype=int),
            "active": pd.Series(active_series, index=test_idx, dtype=bool),
            "eg_pvalue": pd.Series(pval_series, index=test_idx, dtype=float),
        }
    )

    rets_arr = np.array(rets, dtype=float)
    total_return = float(df["equity"].iloc[-1] - 1.0)
    mu_d = float(np.nanmean(rets_arr))
    sd_d = float(np.nanstd(rets_arr, ddof=1)) if len(rets_arr) > 1 else 0.0
    sharpe = (mu_d / sd_d) * np.sqrt(252.0) if sd_d > 0 else 0.0
    # Max drawdown
    roll_max = df["equity"].cummax()
    dd = df["equity"]/roll_max - 1.0
    max_dd = float(dd.min())
    ann_return = float((1.0 + total_return) ** (252.0 / max(len(rets_arr), 1)) - 1.0)

    # Passive benchmark: equal-weight buy-and-hold of the two legs over the OOS
    # window. The strategy is market-neutral, so the honest comparison is
    # risk-adjusted (Sharpe) and whether the neutral book added value beyond
    # simply holding the names.
    bench_ret = 0.5 * (r1 + r2)
    bench_eq = (1.0 + bench_ret).cumprod()
    bench_total = float(bench_eq.iloc[-1] - 1.0) if len(bench_eq) else float("nan")
    bench_sd = float(bench_ret.std(ddof=1)) if len(bench_ret) > 1 else 0.0
    bench_sharpe = (float(bench_ret.mean()) / bench_sd) * np.sqrt(252.0) if bench_sd > 0 else 0.0

    # Plotting (optional) similar to in-sample strategy
    if Graphs == "Y":
        import matplotlib.pyplot as plt
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            # Z-score with signals
            ax1.plot(df.index, df["z"], label="Z-Score", alpha=0.8)
            buys = df[df["signal"] == 1]
            sells = df[df["signal"] == -1]
            ax1.scatter(buys.index, buys["z"], color="green", marker="^", s=80, label="Buy/Close Short")
            ax1.scatter(sells.index, sells["z"], color="red", marker="v", s=80, label="Sell/Close Long")
            ax1.axhline(entry_z, color="red", linestyle="--", alpha=0.6)
            ax1.axhline(-entry_z, color="green", linestyle="--", alpha=0.6)
            ax1.axhline(exit_z, color="orange", linestyle=":", alpha=0.6)
            ax1.axhline(-exit_z, color="orange", linestyle=":", alpha=0.6)
            ax1.axhline(0, color="black", linestyle="-", alpha=0.4)
            ax1.set_title(f"OOS Signals and Z-Score: {s1}/{s2}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Equity curve
            ax2.plot(df.index, df["equity"], label="Equity", linewidth=2)
            ax2.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Start")
            ax2.set_title("Equity Curve (OOS)")
            ax2.set_ylabel("Equity (normalized)")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            if save_plots:
                charts_dir = os.path.join(os.path.dirname(__file__), "charts")
                os.makedirs(charts_dir, exist_ok=True)
                out_path = os.path.join(charts_dir, f"oos_strategy_{s1}_{s2}.png")
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.show()
        except Exception:
            pass

        if debug:
            try:
                fig_dbg, (dx1, dx2, dx3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
                # dx1: Z with ladder thresholds and no-trade band
                dx1.plot(df.index, df["z"], label="Z-Score", color="steelblue", alpha=0.9)
                dx1.axhspan(-exit_z, exit_z, color="orange", alpha=0.15, label="No-trade band")
                z_abs_max = float(np.nanmax(np.abs(df["z"].values))) if len(df) else entry_z
                levels = []
                lvl = float(entry_z)
                while lvl <= z_abs_max + z_step and len(levels) < 50:
                    levels.append(lvl)
                    lvl += float(max(z_step, 1e-8))
                for lvl in levels:
                    dx1.axhline(lvl, color="red", linestyle="--", alpha=0.25)
                    dx1.axhline(-lvl, color="green", linestyle="--", alpha=0.25)
                dx1.axhline(0, color="black", linestyle="-", alpha=0.3)
                dx1.set_ylabel("Z-Score")
                dx1.set_title("Debug: Z with ladder thresholds")
                dx1.grid(True, alpha=0.3)
                dx1.legend(loc="upper right")

                # dx2: Position units and inactive shading
                pos = df["position"].fillna(0)
                dx2.step(df.index, pos, where="post", label="Position (units)", color="purple")
                dx2.axhline(0, color="black", linestyle="--", alpha=0.4)
                active_series_plot = df["active"].fillna(True)
                inactive = (active_series_plot == False)
                if inactive.any():
                    idx = df.index
                    in_seg = False
                    seg_start = None
                    for t, is_inactive in zip(idx, inactive):
                        if is_inactive and not in_seg:
                            in_seg = True
                            seg_start = t
                        elif not is_inactive and in_seg:
                            dx2.axvspan(seg_start, t, color="grey", alpha=0.15, label="Inactive")
                            in_seg = False
                    if in_seg and seg_start is not None:
                        dx2.axvspan(seg_start, idx[-1], color="grey", alpha=0.15)
                dx2.set_ylabel("Units")
                dx2.set_title("Debug: Position units (inactive shaded)")
                dx2.grid(True, alpha=0.3)
                dx2.legend(loc="upper left")

                # dx3: Engle-Granger p-values
                pvals = df["eg_pvalue"]
                dx3.plot(df.index, pvals, label="Engle-Granger p-value", color="brown", alpha=0.8)
                dx3.axhline(stat_sig, color="black", linestyle="--", alpha=0.6, label=f"Threshold {stat_sig}")
                dx3.set_yscale("log")
                dx3.set_ylabel("p-value (log)")
                dx3.set_title("Debug: Engle-Granger p-value at re-tests")
                dx3.grid(True, which="both", alpha=0.3)
                dx3.legend(loc="upper right")

                plt.tight_layout()
                if save_plots:
                    charts_dir = os.path.join(os.path.dirname(__file__), "charts")
                    os.makedirs(charts_dir, exist_ok=True)
                    out_path = os.path.join(charts_dir, f"oos_strategy_{s1}_{s2}_debug.png")
                    plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.show()
            except Exception:
                pass

    return BacktestResult(
        s1=s1,
        s2=s2,
        beta=float(beta_series[-1]) if beta_series else 0.0,
        method="spread",
        entry=entry_z,
        exit=exit_z,
        total_return=total_return,
        ann_return=ann_return,
        sharpe=sharpe,
        max_dd=max_dd,
        trades=trades,
        df=df,
        half_life=hl,
        bench_total_return=bench_total,
        bench_sharpe=bench_sharpe,
    )



def backtest_all_pairs_one_year(
    tickers: List[str],
    split_date: str,
    lookback_years: int = 2,
    entry_z: float = 1.0,
    exit_z: float = 0.2,
    stop_z: float = 4.5,
    tc: float = 0.001,
    Graphs: str = "Y",
    save_plots: bool = True,
    z_step: float = 0.5,
    max_units: int = 5,
    debug: bool = True,
    ignore_gate: bool = False,
    corr_threshold: float = 0.95,
    stat_sig: float = 0.01,
) -> List[Tuple[PairCandidate, BacktestResult]]:
    """Scan all cointegrated pairs in-sample and OOS backtest each for ~1y.

    - In-sample window ends at split_date, length = lookback_years
    - Pairs are filtered by correlation and the Engle-Granger p-value via
      scan_pairs_in_sample
    - For each passing pair, run backtest_pair_one_year on the forward window
    - Prints a concise summary per pair

    If nothing passes the screen, the closest candidates (ranked by p-value)
    are printed as diagnostics and an empty list is returned — an honest "no
    trade" result, not an error.
    """
    split = pd.to_datetime(split_date)
    train_start = (split - pd.DateOffset(years=lookback_years)).date().isoformat()
    panel = fetch_prices(tickers, start=train_start, end=split_date)
    cands = scan_pairs_in_sample(panel, corr_threshold=corr_threshold, stat_sig=stat_sig)
    if not cands:
        near = scan_pairs_in_sample(panel, corr_threshold=corr_threshold, stat_sig=1.0)
        print(f"\nNo pairs passed the in-sample screen "
              f"(corr>={corr_threshold}, Engle-Granger p<{stat_sig}).")
        if near:
            print("Closest candidates (not significant):")
            for pc in near[:5]:
                print(f"  {pc.s1}/{pc.s2} | p={pc.pvalue:.4f} | beta={pc.beta:.3f}")
        else:
            print("No pair even met the correlation threshold; "
                  "widen the universe or lower --corr-threshold.")
        return []

    results: List[Tuple[PairCandidate, BacktestResult]] = []
    print("\n=== In-sample cointegrated pairs (to be OOS tested) ===")
    for i, pc in enumerate(cands, 1):
        print(f"{i:02d}. {pc.s1}/{pc.s2} | method={pc.method} | p={pc.pvalue:.4f} | beta={pc.beta:.3f}")

    print("\n=== Out-of-sample backtest (~252 BDays) per pair ===")
    for i, pc in enumerate(cands, 1):
        try:
            res = backtest_pair_one_year(
                pc.s1,
                pc.s2,
                split_date=split_date,
                lookback_years=lookback_years,
                entry_z=entry_z,
                exit_z=exit_z,
                stop_z=stop_z,
                tc=tc,
                stat_sig=stat_sig,
                Graphs=Graphs,
                save_plots=save_plots,
                z_step=z_step,
                max_units=max_units,
                debug=debug,
                ignore_gate=ignore_gate,
            )
            results.append((pc, res))
            print(
                f"{i:02d}. {pc.s1}/{pc.s2} | p={pc.pvalue:.4f} | half-life={res.half_life:.0f}d | "
                f"total={res.total_return:.2%} (bench {res.bench_total_return:.2%}) | ann={res.ann_return:.2%} | "
                f"sharpe={res.sharpe:.2f} (bench {res.bench_sharpe:.2f}) | maxDD={res.max_dd:.2%} | trades={res.trades}"
            )
        except Exception as exc:
            print(f"{i:02d}. {pc.s1}/{pc.s2} | ERROR during OOS backtest: {exc}")

    return results


if __name__ == "__main__":
    demo_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    split = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
    backtest_all_pairs_one_year(demo_tickers, split_date=split.date().isoformat(), Graphs="Y", save_plots=True)
