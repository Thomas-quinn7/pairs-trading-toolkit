"""Entry point for the pairs-trading toolkit.

This drives the CORRECT, out-of-sample cointegration engine in
``Backtesting.py`` (Engle-Granger OLS hedge ratio, ADF on the residual,
train/test split, one-bar execution lag, quarterly recalibration, and an
out-of-sample cointegration re-validation that disables trading when a pair
stops being cointegrated).

It deliberately does NOT import the legacy ``pair_trader`` code, which had a
full-sample-normalised z-score (look-ahead bias). That flawed v1 is quarantined
in ``legacy/pair_trader_v1_lookahead.py`` and is kept only for the write-up.

Usage:
    python main.py                      # OOS backtest on tickers.txt (or defaults)
    python main.py --no-graphs          # skip plotting
    python main.py --tickers AAPL MSFT  # override the universe
"""

import argparse
import json
import os
import sys
from typing import List, Optional

import pandas as pd

from Backtesting import backtest_all_pairs_one_year


DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def _load_tickers(path: str) -> List[str]:
    """Load tickers from a text/CSV/JSON file (one per line, comma-separated, or JSON list)."""
    if not path or not os.path.exists(path):
        return []
    try:
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [str(x).strip() for x in data if str(x).strip()] if isinstance(data, list) else []
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        items: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.extend([x.strip() for x in line.split(",")] if "," in line else [line])
        return [x for x in items if x]
    except Exception as exc:  # pragma: no cover - defensive file parsing
        print(f"[WARN] Could not read tickers from {path}: {exc}")
        return []


def _dedupe(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Out-of-sample pairs-trading backtest (Engle-Granger + OOS re-validation)"
    )
    default_file = os.path.join(os.path.dirname(__file__), "tickers.txt")
    ap.add_argument("--tickers", nargs="*", default=[], help="Tickers to analyse (overrides the file)")
    ap.add_argument("--tickers-file", default=default_file, help="File of tickers (one per line, CSV, or JSON list)")

    ap.add_argument("--split-date", default=None,
                    help="Train/test boundary (YYYY-MM-DD). Default: one year ago. "
                         "Training uses the lookback window ending here; the OOS test runs forward from it.")
    ap.add_argument("--lookback-years", type=int, default=2, help="Training window length in years (default 2)")
    ap.add_argument("--entry-z", type=float, default=1.0, help="Z-score entry threshold")
    ap.add_argument("--exit-z", type=float, default=0.2, help="Z-score exit threshold")
    ap.add_argument("--stop-z", type=float, default=4.5, help="Z-score stop-loss threshold")
    ap.add_argument("--tc", type=float, default=0.001, help="Transaction cost per trade (fraction)")
    ap.add_argument("--no-graphs", action="store_true", help="Disable plotting")
    ap.add_argument("--no-save-plots", action="store_true", help="Do not write plots to charts/")

    return ap.parse_args(argv or [])


def main(argv: Optional[List[str]] = None) -> int:
    ns = parse_args(argv)

    # Resolve the universe: explicit CLI tickers > file > built-in defaults.
    if ns.tickers:
        tickers = _dedupe([str(t).strip() for t in ns.tickers])
        print(f"[INFO] Using {len(tickers)} tickers from CLI")
    else:
        tickers = _dedupe(_load_tickers(ns.tickers_file))
        if tickers:
            print(f"[INFO] Loaded {len(tickers)} tickers from {ns.tickers_file}")
        else:
            tickers = DEFAULT_TICKERS
            print(f"[INFO] Using built-in default tickers: {', '.join(tickers)}")

    # Default split = one year ago, giving ~1y of out-of-sample data forward.
    if ns.split_date:
        split_date = ns.split_date
    else:
        split_date = (pd.Timestamp.today().normalize() - pd.Timedelta(days=365)).date().isoformat()
    print(f"[INFO] Train/test split date: {split_date} (lookback {ns.lookback_years}y, ~252 BDay OOS window)")

    print("[INFO] Running out-of-sample cointegration backtest via Backtesting.py ...")
    try:
        backtest_all_pairs_one_year(
            tickers,
            split_date=split_date,
            lookback_years=ns.lookback_years,
            entry_z=ns.entry_z,
            exit_z=ns.exit_z,
            stop_z=ns.stop_z,
            tc=ns.tc,
            # Backtesting.py gates plotting on the string "Y"; pass it explicitly.
            Graphs="N" if ns.no_graphs else "Y",
            save_plots=not ns.no_save_plots,
        )
    except Exception as exc:
        print(f"[ERROR] Backtest failed: {exc}")
        return 1

    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
