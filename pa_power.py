#!/usr/bin/env python3
"""
pa_power.py -- how big does a real edge have to be before my verifier
can see it?

The challenge this answers, in Dante's words: "I think I could give you
a proven strategy and you'd still say it doesn't work."

That is a claim about the POWER of the test, and power is measurable.
The method: build strategies on real bars whose edge size I control
exactly, push each one through the IDENTICAL pipeline used to reject the
Playbook, and count how often the pipeline finds it.

HOW AN EDGE OF KNOWN SIZE IS PLANTED. Entries are at random bars, so
every trade uses real prices, real exits and real costs -- nothing is
synthetic except WHICH bars get traded. A candidate trade is accepted
with a probability that depends on its own outcome: all winners kept,
losers kept with probability q. Lower q, more foresight, bigger edge.
That is a strategy with genuine predictive skill of a dialled-in
amount. It is not a backtest of anything tradeable -- it is a ruler.

WHAT COMES OUT is a power curve: for a given true edge and a given
number of trades, the share of runs where the verifier says yes. Read it
as the honest statement of what I can and cannot detect:
  - above the detection floor, a real edge gets found
  - below it, I will reject real edges, and Dante is right

TWO GATES ARE MEASURED SEPARATELY, because they failed differently on
this project:
  t > 1.96   on R per trade -- the gate that rejected the Playbook
  null p<.05 beats randomly-timed trades -- the gate used throughout

USAGE
    python3 pa_power.py
    python3 pa_power.py --selftest
"""
from __future__ import annotations

import argparse
import logging
import zlib

import numpy as np
import pandas as pd

log = logging.getLogger("pa_power")

STOP_R = 0.03        # 3% stop: wide enough to be fillable on 12h bars
TARGET_R = 2.0       # 2R target, matching the Playbook tests
ROUND_TRIP = 0.0017  # same cost assumption as pa_book
MAX_HOLD = 60
REPS = 200           # runs per (edge, n) cell


def trade(h, l, c, eb, side):
    """One trade on real bars: fixed-fraction stop, 2R target.

    A bar touching both stop and target is read as the stop -- the
    conservative reading when only OHLC is known.
    """
    n = len(c)
    if eb >= n - 1:
        return None
    ent = c[eb]
    if not np.isfinite(ent) or ent <= 0:
        return None
    stop = ent * (1 - side * STOP_R)
    tgt = ent * (1 + side * TARGET_R * STOP_R)
    for i in range(eb + 1, min(eb + MAX_HOLD, n)):
        if side > 0 and l[i] <= stop:
            return -STOP_R - ROUND_TRIP
        if side < 0 and h[i] >= stop:
            return -STOP_R - ROUND_TRIP
        if side > 0 and h[i] >= tgt:
            return TARGET_R * STOP_R - ROUND_TRIP
        if side < 0 and l[i] <= tgt:
            return TARGET_R * STOP_R - ROUND_TRIP
    last = min(eb + MAX_HOLD, n) - 1
    return side * (c[last] / ent - 1) - ROUND_TRIP


def pool(coins, tf, loader):
    """Pre-compute every candidate trade once, for both sides.

    Doing this up front means the sampling loop never touches price data
    again, which is what makes 200 reps x several cells tractable.
    """
    out = []
    for coin in coins:
        df = loader(coin, tf)
        if df is None or len(df) < 300:
            continue
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        for side in (1, -1):
            for eb in range(50, len(c) - MAX_HOLD - 2):
                p = trade(h, l, c, eb, side)
                if p is not None and np.isfinite(p):
                    out.append((p, side))
    a = np.array(out, dtype=float)
    return a[:, 0], a[:, 1].astype(int)


def plant(pnl, win, q, n, rg):
    """Draw n trades, keeping all winners and losers with prob q."""
    keep = np.where(win, True, rg.random(len(pnl)) < q)
    idx = np.flatnonzero(keep)
    if len(idx) < n:
        return None
    return pnl[rg.choice(idx, size=n, replace=False)]


def null_mean(pnl, n, rg, sims=40):
    """The verifier's null: n randomly-timed trades, same machinery."""
    return np.array([pnl[rg.integers(0, len(pnl), n)].mean()
                     for _ in range(sims)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=12)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import event_audit as EA
    import short_risk as SR
    rg = np.random.default_rng(zlib.crc32(b"pa_power"))
    coins = SR.universe()
    pnl, side = pool(coins, a.tf, EA.load_bars)
    win = pnl > 0
    base = pnl.mean() / STOP_R
    print(f"\n  candidate trades: {len(pnl):,} on {a.tf}h bars, "
          f"{STOP_R*100:.0f}% stop, {TARGET_R:.0f}R target")
    print(f"  base win rate {win.mean()*100:.1f}%, "
          f"base edge {base:+.3f}R  (this is the no-skill baseline)\n")

    if a.selftest:
        # q=1 keeps everything: must reproduce the baseline, no edge
        s = plant(pnl, win, 1.0, 4000, rg)
        e = s.mean() / STOP_R
        print(f"  q=1.0 (no skill): {e:+.3f}R vs base {base:+.3f}R")
        assert abs(e - base) < 0.05, "planting with q=1 changed the edge"
        # lower q must raise the edge monotonically
        es = [plant(pnl, win, q, 4000, rg).mean() / STOP_R
              for q in (1.0, 0.8, 0.6, 0.4)]
        print("  edge as q falls:", " ".join(f"{x:+.3f}" for x in es))
        assert all(x < y for x, y in zip(es, es[1:])), "not monotonic"
        print("  the ruler is calibrated: OK")
        return 0

    qs = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60]
    ns = [100, 300, 1000]
    print(f"  detection rate over {a.reps} runs per cell")
    print(f"  {'true edge':>10}{'win%':>7}   "
          + "".join(f"{'n='+str(n):>21}" for n in ns))
    print(f"  {'':>10}{'':>7}   " + "".join(f"{'t>1.96':>10}{'null':>11}"
                                            for _ in ns))
    for q in qs:
        s = plant(pnl, win, q, min(20000, len(pnl)), rg)
        true_edge = s.mean() / STOP_R
        true_win = (s > 0).mean()
        cells = []
        for n in ns:
            ht = hn = 0
            for _ in range(a.reps):
                smp = plant(pnl, win, q, n, rg)
                if smp is None:
                    continue
                R = smp / STOP_R
                se = R.std(ddof=1) / np.sqrt(n)
                if se > 0 and R.mean() / se > 1.96:
                    ht += 1
                nl = null_mean(pnl, n, rg)
                if (nl >= smp.mean()).mean() < 0.05:
                    hn += 1
            cells.append((ht / a.reps, hn / a.reps))
        print(f"  {true_edge:>+9.3f}R{true_win*100:>6.1f}%   "
              + "".join(f"{t*100:>9.0f}%{nn*100:>10.0f}%"
                        for t, nn in cells))
    print("\n  Read the floor: the smallest true edge my verifier finds")
    print("  most of the time. Below it I reject real edges -- so if a")
    print("  strategy's true edge is under that, Dante is right and the")
    print("  answer is more data, not a different verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
