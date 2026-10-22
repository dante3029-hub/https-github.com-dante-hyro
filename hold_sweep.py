#!/usr/bin/env python3
"""
hold_sweep.py -- does a longer hold keep the signal and drop the fees?

THE OBSERVATION, which came from the account owner looking at the economics
rather than the Sharpe: the book takes months to make 10% and pays fees the
whole way. Measured, scale-invariantly, as a share of each sleeve's own
notional:

    sleeve   ann vol   net ret   fees/yr   gross   fees as % of gross
    relvol     14.2%     15.5%      6.6%   22.2%          30%
    oirank      8.6%      7.7%     15.5%   23.2%          67%     <--

oirank rebalances every 3 DAYS -- 122 times a year. Two thirds of its gross
edge is paid to the exchange, and nobody ever questioned the 3.

If the SIGNAL survives a longer hold, the fee saving alone is worth more
than any new sleeve we have found:

    hold  3d  fees 15.5%  -> Sharpe 0.90   (current)
    hold  5d  fees  9.3%  -> Sharpe 1.62   IF the signal is unchanged
    hold  7d  fees  6.6%  -> Sharpe 1.93   IF the signal is unchanged
    hold 10d  fees  4.7%  -> Sharpe 2.16   IF the signal is unchanged

Those are arithmetic, not results. The signal will NOT be unchanged. A
3-day signal held 10 days decays -- the question is whether it decays more
slowly than the fees fall. This file measures that instead of assuming it.

PRE-REGISTERED, fixed before the first run:
    sleeves   oirank, relvol
    holds     3, 5, 7, 10, 14
    n         unchanged from live (oirank 5, relvol 8)
    universe  unchanged (CORE24)
    10 configs total, counted in the Bonferroni

WHY THIS IS NOT A FISHING EXPEDITION, which matters given that a hold sweep
is exactly the shape of a fitted parameter: there is a mechanical, a-priori
reason to expect improvement that has nothing to do with the data -- fees
fall as 1/hold, exactly and knowably. The null hypothesis is specific: the
signal decays at least as fast as the fees fall, so Sharpe does not rise.

A WARNING FROM THIS SLEEVE'S OWN HISTORY. sleeve_oirank's docstring records
that a single 7-day LOOKBACK scored 1.41 against ~0.95 at 5d and 10d -- a
45% spike that was correctly diagnosed as fitted and blended away. If the
HOLD sweep shows the same shape (one spike, neighbours much lower), it is
the same disease and must be treated the same way. A real effect is a
PLATEAU. Read the shape, not the maximum.

USAGE
    python3 -u hold_sweep.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import book
from book import FEE, stats, xs

TURN = 1.5          # measured: ~1.5x gross changes hands per rebalance
HOLDS = (3, 5, 7, 10, 14)


def log(m=""):
    print(m, flush=True)


def main() -> int:
    log("  loading panel...")
    idx, PX, R, VOL, DN = book.panel()
    log(f"  {len(R.columns)} coins, {len(idx)} days\n")

    Z = lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x * 0.0
    cfgs = {}

    # relvol: exact live construction from book.build()
    cfgs["relvol"] = (lambda t: Z(VOL.iloc[t - 1] / VOL.iloc[t - 21:t - 1].mean()), 8)

    # oirank: needs OI, and the t-1 adapter (see tilt_audit for why)
    try:
        from tilt_audit import load_oi_daily
        from signal_engine.sleeve_oirank import oi_scores
        OIdf, note = load_oi_daily(list(R.columns), idx)
        if OIdf is not None:
            OI = OIdf.values
            cfgs["oirank"] = (
                lambda t: pd.Series(oi_scores(OI, t - 1), index=R.columns), 5)
            log(f"  oirank OI: {note}")
        else:
            log(f"  oirank SKIPPED: {note}")
    except Exception as e:
        log(f"  oirank SKIPPED: {type(e).__name__}: {e}")

    log("")
    log(f"  {'sleeve':<9}{'hold':>6}{'Sharpe':>9}{'ann':>8}{'1st':>7}{'2nd':>7}"
        f"{'both':>6}{'fees/yr':>9}")
    results = {}
    for name, (sig, n) in cfgs.items():
        for h in HOLDS:
            s = xs(R, sig, n=n, hold=h)
            sh, ann = stats(s)
            hm = len(s) // 2
            s1 = stats(s.iloc[:hm])[0]
            s2 = stats(s.iloc[hm:])[0]
            fees = TURN * FEE * (365 / h) * 100
            both = "yes" if s1 > 0 and s2 > 0 else "NO"
            cur = "  <- live" if (name == "oirank" and h == 3) or \
                                 (name == "relvol" and h == 7) else ""
            log(f"  {name:<9}{h:>5}d{sh:>9.2f}{ann:>7.1f}%{s1:>7.2f}{s2:>7.2f}"
                f"{both:>6}{fees:>8.1f}%{cur}")
            results[(name, h)] = (sh, s1, s2)
        log("")

    # --- shape, not maximum -------------------------------------------------
    log("  " + "=" * 66)
    log("  SHAPE CHECK -- a real effect is a PLATEAU, a fitted one is a SPIKE")
    log("  " + "=" * 66)
    for name in cfgs:
        v = [results[(name, h)][0] for h in HOLDS]
        best_i = int(np.argmax(v))
        best_h = HOLDS[best_i]
        nb = [v[i] for i in (best_i - 1, best_i + 1) if 0 <= i < len(v)]
        log(f"    {name}: best {best_h}d at {v[best_i]:.2f}; "
            f"neighbours {', '.join(f'{x:.2f}' for x in nb)}")
        if v[best_i] <= 0:
            # Comparing negative Sharpes as a ratio is meaningless -- the
            # "spike" would just be which loss is least bad. Say so instead
            # of printing a confident verdict about a sleeve that loses at
            # every hold.
            log(f"      NO EDGE AT ANY HOLD -- best is {v[best_i]:.2f}. The")
            log(f"      shape test does not apply; there is nothing to shape.")
            log("")
            continue
        # gap measured against the better neighbour, in Sharpe points as well
        # as percent, because a percentage of a small number flatters itself
        gap_pts = v[best_i] - max(nb)
        gap = gap_pts / abs(max(nb)) * 100 if nb and max(nb) > 0 else 0.0
        if gap > 30 and gap_pts > 0.25:
            log(f"      SPIKE: {gap:.0f}% (+{gap_pts:.2f}) above its neighbours. Same")
            log(f"      shape the 7-day LOOKBACK had (1.41 vs ~0.95) and that")
            log(f"      was correctly judged FITTED. Do not take the maximum.")
        else:
            log(f"      plateau ({gap:+.0f}%, {gap_pts:+.2f} vs neighbours) -- what a real")
            log(f"      effect makes. Safe to prefer the cheaper hold.")
        halves_ok = all(results[(name, h)][1] > 0 and results[(name, h)][2] > 0
                        for h in HOLDS if results[(name, h)][0] > 0)
        log(f"      halves positive across all positive holds: "
            f"{'yes' if halves_ok else 'NO -- regime-dependent'}")
        log("")

    log("  Pick the LONGEST hold whose Sharpe is within noise of the best,")
    log("  not the best. Longer is cheaper, and cheaper is the one thing")
    log("  here that is certain rather than estimated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
