import argparse
import sys
from typing import List, Optional

import os
import json
from datetime import datetime
import pandas as pd


# Support both package execution (python -m Pairs_trading_tools.main)
# and direct script execution (python Pairs_trading_tools/main.py)
try:
    from .pair_trader import (
        data_fetcher,
        heatmap,
        coint_tester,
        strat_stats,
        moving_average_strategy,
    )
except Exception:  # pragma: no cover - fallback for direct script run
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from Pairs_trading_tools.pair_trader import (  # type: ignore
        data_fetcher,
        heatmap,
        coint_tester,
        strat_stats,
        moving_average_strategy,
    )


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_pairs(pairs_df: pd.DataFrame, out_csv: str) -> None:
    ensure_dir(out_csv)
    # Drop the large embedded price series when saving summary
    to_save = pairs_df.copy()
    for col in ["stock_1_data", "stock_2_data"]:
        if col in to_save.columns:
            to_save[col] = to_save[col].apply(lambda s: getattr(s, "name", "series"))
    to_save.to_csv(out_csv, index=False)


def load_pairs(in_csv: str) -> pd.DataFrame:
    df = pd.read_csv(in_csv)
    # Placeholder for rehydration if needed; currently only names are stored.
    return df


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pairs trading CLI orchestrator")
    default_file = os.path.join(os.path.dirname(__file__), "tickers.txt")
    ap.add_argument("--tickers", nargs="*", default=[], help="Tickers to analyze (overrides file if provided)")
    ap.add_argument("--tickers-file", default=default_file, help="Path to a file containing tickers (one per line, CSV, or JSON list)")

    # Tasks
    ap.add_argument("--heatmap", action="store_true", help="Generate and save correlation heatmap")
    ap.add_argument("--scan", action="store_true", help="Scan and print cointegrated pairs")
    ap.add_argument("--stats", action="store_true", help="Plot ratio/z-score for a chosen pair index from scan")
    ap.add_argument("--ma-strat", action="store_true", help="Run moving-average strategy on chosen pair index from scan")
    ap.add_argument("--pipeline", action="store_true", help="Run heatmap + scan + (optional) stats/strategy in one go (default)")

    # Scan options
    ap.add_argument("--corr-threshold", type=float, default=0.9, help="Correlation threshold for scanning")
    ap.add_argument("--stat-sig", type=float, default=0.01, help="ADF p-value significance level (1% default)")
    ap.add_argument("--pairs-out", default=os.path.join(os.path.dirname(__file__), "charts", "coint_pairs.csv"), help="Path to save pairs scan results")
    ap.add_argument("--pairs-in", default=None, help="Optional path to pre-saved pairs to skip scanning")

    # Selection and strategy params
    ap.add_argument("--pair-index", type=int, default=0, help="Index of pair in scan results to use")
    ap.add_argument("--ma-short", type=int, default=5, help="Short MA window for strategy")
    ap.add_argument("--ma-long", type=int, default=15, help="Long MA window for strategy")
    ap.add_argument("--z-entry", type=float, default=0.5, help="Z-score entry threshold")
    ap.add_argument("--z-exit", type=float, default=0.1, help="Z-score exit threshold")
    ap.add_argument("--initial-capital", type=float, default=10000.0, help="Initial capital for backtest")
    ap.add_argument("--transaction-cost", type=float, default=0.001, help="Cost per trade (fraction)")
    ap.add_argument("--stop-loss-z", type=float, default=4.5, help="Emergency z-score stop loss")
    ap.add_argument("--stop-loss-ratio", type=float, default=0.3, help="Emergency ratio change stop loss")
    ap.add_argument("--no-graphs", action="store_true", help="Disable graphs in strategy run")
    ap.add_argument("--no-performance", action="store_true", help="Disable performance printouts in strategy run")
    ap.add_argument("--all-pairs", action="store_true", help="Apply stats/strategy to all scanned pairs instead of a single index")

    return ap.parse_args(argv or [])


def main(argv: Optional[List[str]] = None) -> int:
    ns = parse_args(argv)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Decide pipeline default if no explicit task flags
    run_pipeline = ns.pipeline or not (ns.heatmap or ns.scan or ns.stats or ns.ma_strat)

    pairs_df: Optional[pd.DataFrame] = None

    # Resolve tickers from file or CLI
    tickers: List[str] = []
    def _dedupe(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    def _load_tickers(path: str) -> List[str]:
        if not os.path.exists(path):
            return []
        try:
            # JSON list
            if path.lower().endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()]
                return []
            # Text/CSV
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            items: List[str] = []
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "," in line:
                    items.extend([x.strip() for x in line.split(",")])
                else:
                    items.append(line)
            return [x for x in items if x]
        except Exception:
            return []

    file_tickers = _load_tickers(ns.tickers_file) if ns.tickers_file else []
    if file_tickers:
        tickers = _dedupe(file_tickers)
        print(f"[INFO] Loaded {len(tickers)} tickers from {ns.tickers_file}")
    elif ns.tickers:
        tickers = _dedupe([str(t).strip() for t in ns.tickers if str(t).strip()])
        print(f"[INFO] Using {len(tickers)} tickers from CLI")
    else:
        # Fallback defaults if nothing provided
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        print("[INFO] Using built-in default tickers (AAPL, MSFT, GOOGL, AMZN, META)")

    if ns.heatmap or run_pipeline:
        print("[INFO] Generating correlation heatmap...")
        try:
            heatmap(tickers)
            print("[OK] Heatmap saved as heatmap.png")
        except Exception as exc:
            print(f"[WARN] Heatmap failed: {exc}")

    if ns.pairs_in and os.path.exists(ns.pairs_in):
        print(f"[INFO] Loading pairs from {ns.pairs_in}")
        pairs_df = load_pairs(ns.pairs_in)
    if ns.scan or run_pipeline or pairs_df is None:
        print("[INFO] Scanning for cointegrated pairs...")
        try:
            pairs_df = coint_tester(
                tickers,
                corr_threshold=ns.corr_threshold,
                Output_adfuller=True,
                stat_significant=ns.stat_sig,
            )
            print(pairs_df[["s1", "s2", "pvs", "pvr"]])
            if ns.pairs_out:
                save_path = ns.pairs_out
                # Ensure directory exists
                ensure_dir(save_path)
                save_pairs(pairs_df, save_path)
                print(f"[OK] Saved pairs to {save_path}")
        except Exception as exc:
            print(f"[WARN] Scan failed: {exc}")
            pairs_df = None

    if (ns.stats or run_pipeline) and pairs_df is not None and len(pairs_df) > 0:
        if ns.all_pairs:
            print(f"[INFO] Plotting ratio/z for all {len(pairs_df)} pairs")
            for idx in range(len(pairs_df)):
                try:
                    strat_stats(pairs_df, item=idx)
                except Exception as exc:
                    print(f"[WARN] Stats plotting failed for pair index {idx}: {exc}")
        else:
            print(f"[INFO] Plotting ratio/z for pair index {ns.pair_index}")
            try:
                strat_stats(pairs_df, item=ns.pair_index)
            except Exception as exc:
                print(f"[WARN] Stats plotting failed: {exc}")

    if (ns.ma_strat or run_pipeline) and pairs_df is not None and len(pairs_df) > 0:
        if ns.all_pairs:
            print(f"[INFO] Running moving-average strategy for all {len(pairs_df)} pairs")
            for idx in range(len(pairs_df)):
                try:
                    moving_average_strategy(
                        pairs_df,
                        item=idx,
                        ma_short=ns.ma_short,
                        ma_long=ns.ma_long,
                        z_entry=ns.z_entry,
                        z_exit=ns.z_exit,
                        initial_capital=ns.initial_capital,
                        transaction_cost=ns.transaction_cost,
                        stop_loss_z=ns.stop_loss_z,
                        stop_loss_ratio=ns.stop_loss_ratio,
                        Performance="N" if ns.no_performance else "Y",
                        Graphs="N" if ns.no_graphs else "Y",
                        save_plots=True,
                    )
                except Exception as exc:
                    print(f"[WARN] Strategy run failed for pair index {idx}: {exc}")
        else:
            print(f"[INFO] Running moving-average strategy for pair index {ns.pair_index}")
            try:
                moving_average_strategy(
                    pairs_df,
                    item=ns.pair_index,
                    ma_short=ns.ma_short,
                    ma_long=ns.ma_long,
                    z_entry=ns.z_entry,
                    z_exit=ns.z_exit,
                    initial_capital=ns.initial_capital,
                    transaction_cost=ns.transaction_cost,
                    stop_loss_z=ns.stop_loss_z,
                    stop_loss_ratio=ns.stop_loss_ratio,
                    Performance="N" if ns.no_performance else "Y",
                    Graphs="N" if ns.no_graphs else "Y",
                    save_plots=True,
                )
            except Exception as exc:
                print(f"[WARN] Strategy run failed: {exc}")

    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
