"""Quantify the look-ahead bias that was in the v1 engine.

The quarantined v1 strategy (legacy/pair_trader_v1_lookahead.py) computed its
z-score with the FULL-SAMPLE mean and standard deviation:

    z = (spread - spread.mean()) / spread.std()

so every historical signal already "knew" the spread's future distribution -
in particular where the mean was. This script runs the *same* mean-reversion
rule two ways on many synthetic Ornstein-Uhlenbeck spreads:

  * full-sample z  - look-ahead, as in v1
  * trailing z     - causal, using only data up to each bar, as in Backtesting.py

and reports the Sharpe of each. The full-sample version reports a systematically
better Sharpe that is not achievable in real time. That gap is the bias.

No network, no third-party data. Run:  python demo_lookahead.py
"""

from __future__ import annotations

import numpy as np


def ou_spread(rng, n=750, phi=0.95, sd=1.0):
    """A stationary AR(1)/OU spread."""
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = phi * z[t - 1] + rng.normal(0.0, sd)
    return z


def strategy_sharpe(spread, z, entry=1.0, exit=0.2):
    """Long the spread when z<-entry, short when z>entry, flat near the mean.

    Position is applied with a one-bar delay; P&L is position * change in spread.
    Returns the annualised Sharpe of the daily P&L.
    """
    n = len(spread)
    pos = np.zeros(n)
    for t in range(1, n):
        p = pos[t - 1]
        if abs(z[t]) <= exit:
            p = 0.0
        elif z[t] > entry:
            p = -1.0
        elif z[t] < -entry:
            p = +1.0
        pos[t] = p
    dspread = np.diff(spread, prepend=spread[0])
    pnl = pos[:-1] * dspread[1:]           # one-bar delay: yesterday's position
    sd = pnl.std(ddof=1)
    return (pnl.mean() / sd) * np.sqrt(252.0) if sd > 0 else 0.0


def full_sample_z(spread):
    return (spread - spread.mean()) / spread.std()   # LOOK-AHEAD


def trailing_z(spread, min_periods=60):
    """Causal z: mean/std from data up to and including each bar only."""
    z = np.zeros_like(spread)
    for t in range(len(spread)):
        if t + 1 < min_periods:
            z[t] = 0.0
            continue
        window = spread[: t + 1]
        sd = window.std()
        z[t] = (spread[t] - window.mean()) / sd if sd > 0 else 0.0
    return z


def main():
    n_paths = 400
    # Regimes from lots of data / weakly persistent to little data / very
    # persistent - the realistic case for a real pair with limited history.
    regimes = [
        (750, 0.95, "3.0y  phi=0.95"),
        (500, 0.97, "2.0y  phi=0.97"),
        (250, 0.98, "1.0y  phi=0.98"),
        (150, 0.98, "0.6y  phi=0.98"),
    ]

    print("Look-ahead bias in the pairs z-score")
    print(f"  {n_paths} synthetic OU spreads per regime, identical mean-reversion rule.")
    print("  full-sample z = v1 (look-ahead) ; trailing z = Backtesting.py (causal).\n")
    print(f"  {'data / persistence':<18}{'full-sample':>13}{'trailing':>12}{'inflation':>18}")
    for n, phi, label in regimes:
        rng = np.random.default_rng(2024)  # same draws across regimes for fairness
        la, honest = [], []
        for _ in range(n_paths):
            s = ou_spread(rng, n=n, phi=phi)
            la.append(strategy_sharpe(s, full_sample_z(s)))
            honest.append(strategy_sharpe(s, trailing_z(s)))
        la_m, ho_m = float(np.mean(la)), float(np.mean(honest))
        infl = f"+{la_m - ho_m:.2f} ({la_m / ho_m:.1f}x)" if ho_m > 0 else "n/a"
        print(f"  {label:<18}{la_m:>+13.2f}{ho_m:>+12.2f}{infl:>18}")

    print("\n  The full-sample z reports a Sharpe that cannot be earned in real time -")
    print("  it is reading the future mean and vol of the spread. The gap widens as")
    print("  history shrinks and the spread gets more persistent, i.e. exactly the")
    print("  regime real pairs live in. Backtesting.py uses the trailing/train-window")
    print("  version, which is all a live desk actually has.")


if __name__ == "__main__":
    main()
