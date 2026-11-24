#!/usr/bin/env python3
"""
pa_score.py -- the Playbook's SELECTION layer, which pa_book.py skipped.

pa_book.py built the trigger and took every range it found: 17,000
trades. Matt takes two or three a week on one chart. The gap is Part 3
of the book, which is entirely about deciding which levels are worth
trading, and it was missing. A small edge spread across every mediocre
setup is exactly what unstable halves look like.

THE FOUR FILTERS, each quoted from the book so the intent is auditable:

  LOCATION (p.53) "I only trade the ones at swing highs, at swing lows
      or at the origin of a move that broke a barrier. The ones in the
      middle of a swing are way less reliable."
      -> the range must sit at a confirmed swing extreme, or be a
         DECISION POINT.

  DECISION POINT (p.44) "the origin of a move that goes on to take out a
      barrier". The move out of the range must break a PRIOR swing
      level. That is what proves volume was behind it, and it is the
      difference between a zone someone drew and a zone that did work.

  MAX PAIN (p.48) supply: highest wick to highest close; demand: lowest
      wick to lowest close. Entry at the extreme of the zone, not its
      edge -- tighter stop, so the same move is worth more R.

  TREND CONTEXT (p.59) "take continuations after the first impulse and
      not after a massive move has already happened."
      -> count impulses since the trend began; skip the late ones.

EVERY THRESHOLD IS STILL FIXED, NOT TUNED. swing_k=2 and the momentum
multiple carry over from pa_book. The two new numbers -- how far back to
look for a barrier, and how many impulses count as "extended" -- are set
once here and never swept. Four filters x a few plausible values each
would be another 30-cell grid, and this project has already shown what
those produce out of sample.

WHAT WOULD MAKE THIS REAL: the trade count collapsing AND the edge per
trade rising. If filtering cuts 17,000 to 300 and the edge stays the
same, the filters are just sampling and the book's selectivity is
decoration. If the edge concentrates, it is doing what he says.

USAGE
    python3 pa_score.py --tf 12
    python3 pa_score.py --tf 12 --selftest
"""
from __future__ import annotations

import argparse
import logging
import zlib

import numpy as np
import pandas as pd

import pa_book as P

log = logging.getLogger("pa_score")

BARRIER_LOOKBACK = 30      # bars back to find a prior swing the move broke
MAX_IMPULSES = 2           # continuations only after impulse 1 or 2
SWING_TOL = 0.002          # 0.2% -- "at" a swing extreme, not exactly on it
ZONE_TOUCH = 0.25          # a bar "at" the zone: within 25% of range width


# --------------------------------------------------------------- filters
def at_swing_extreme(rng: P.Rng, h, l, sh, sl, tol=SWING_TOL) -> bool:
    """Is this range sitting at a confirmed swing high or low?

    Checked against swings CONFIRMED BEFORE the range went live
    (rng.start), never the ones that formed afterwards -- otherwise the
    filter is reading the future and every result after it is void.
    """
    lo_i = max(0, rng.start - BARRIER_LOOKBACK)
    idx = [i for i in range(lo_i, rng.start) if sh[i] or sl[i]]
    for i in idx:
        if P.confirmed_at(i) > rng.start:
            continue                      # not knowable yet
        if sh[i] and abs(h[i] - rng.hi) / max(rng.hi, 1e-12) < tol:
            return True
        if sl[i] and abs(l[i] - rng.lo) / max(rng.lo, 1e-12) < tol:
            return True
    return False


def is_decision_point(rng: P.Rng, h, l, c, sh, sl) -> bool:
    """Did the move OUT of this range break a prior swing level?

    The book's definition: the origin of a move that takes out a
    barrier. Looks only at swings formed BEFORE the range, and only at
    price action from the range forward, so nothing here is hindsight
    about the trade itself -- the break has to have happened before the
    entry bar for the range to qualify.
    """
    lo_i = max(0, rng.start - BARRIER_LOOKBACK)
    prior_h = [h[i] for i in range(lo_i, rng.start)
               if sh[i] and P.confirmed_at(i) <= rng.start]
    prior_l = [l[i] for i in range(lo_i, rng.start)
               if sl[i] and P.confirmed_at(i) <= rng.start]
    end = min(rng.start + P.MAX_WAIT, len(c))
    seg = c[rng.start:end]
    if seg.size == 0:
        return False
    if rng.trend_up and prior_h:
        return bool(seg.max() > max(prior_h))
    if (not rng.trend_up) and prior_l:
        return bool(seg.min() < min(prior_l))
    return False


def max_pain(rng: P.Rng, o, h, l, c, supply: bool):
    """Refine the zone to max pain (p.48).

    supply zone: highest wick -> highest close
    demand zone: lowest wick  -> lowest close

    FIRST VERSION OF THIS WAS WRONG and the mistake is worth keeping on
    the record. It took the extremes from a fixed 30-bar lookback and
    moved only the STOP, leaving the entry at the breakout bar. That
    made the median stop 17.9% wide against 12.4% unfiltered -- the
    exact opposite of the book, whose entire reason for finding max pain
    is a TIGHTER stop. It measured a bug, not the method.

    The zone belongs to the RANGE's own extreme: the highest wick of a
    supply zone IS rng.hi by construction, and the zone's inner edge is
    the highest CLOSE among the bars that printed up there. Entry is a
    limit at that inner edge; the stop sits just past the wick.

    Returns (entry_level, stop_beyond) or None if degenerate.
    """
    b = rng.start
    a = max(0, b - BARRIER_LOOKBACK)
    if b - a < 3:
        return None
    if supply:
        wick = float(rng.hi)
        # bars that actually traded at the zone's high
        near = [c[i] for i in range(a, b)
                if h[i] >= wick - ZONE_TOUCH * (rng.hi - rng.lo)]
        if not near:
            return None
        level = float(max(near))
        if not (wick > level > 0):
            return None
        return level, wick
    wick = float(rng.lo)
    near = [c[i] for i in range(a, b)
            if l[i] <= wick + ZONE_TOUCH * (rng.hi - rng.lo)]
    if not near:
        return None
    level = float(min(near))
    if not (0 < wick < level):
        return None
    return level, wick


def simulate_limit(o, h, l, c, eb, side, level, beyond, mode,
                   wait=None, hold=None):
    """The book's actual entry: a LIMIT at the zone, not a market fill.

    Price must come back and trade through `level` within MAX_WAIT bars
    of the signal. Fill is at `level`; the stop sits just past `beyond`.
    A bar that reaches both the fill and the stop is read as filled THEN
    stopped -- the conservative assumption when only OHLC is known.

    No fill means no trade. That is the book's selectivity doing its
    work: most signals never get the retest, so they never cost
    anything.
    """
    n = len(c)
    wait = P.MAX_WAIT if wait is None else wait
    hold = P.MAX_HOLD if hold is None else hold
    r = abs(level - beyond) / level
    if r <= 0 or not np.isfinite(r):
        return None
    tgt = level * (1 + side * P.MIN_R * r)
    fill = None
    for i in range(eb, min(eb + wait, n)):
        if side < 0 and h[i] >= level:
            fill = i
            break
        if side > 0 and l[i] <= level:
            fill = i
            break
    if fill is None:
        return None
    for i in range(fill, min(fill + hold, n)):
        if side > 0 and l[i] <= beyond:
            return (beyond / level - 1) - P.ROUND_TRIP, i - fill, r
        if side < 0 and h[i] >= beyond:
            return (1 - beyond / level) - P.ROUND_TRIP, i - fill, r
        if side > 0 and h[i] >= tgt:
            return (tgt / level - 1) - P.ROUND_TRIP, i - fill, r
        if side < 0 and l[i] <= tgt:
            return (1 - tgt / level) - P.ROUND_TRIP, i - fill, r
    # timed out flat
    last = min(fill + hold, n) - 1
    return (side * (c[last] / level - 1)) - P.ROUND_TRIP, last - fill, r


def impulse_count(rng: P.Rng, c, sh, sl) -> int:
    """How many impulses deep into the trend is this range? (p.59)

    Counted as confirmed swings in the trend direction before the range.
    One or two means early; more means the move is extended and the
    book says leave it alone.
    """
    lo_i = max(0, rng.start - BARRIER_LOOKBACK * 2)
    flags = sh if rng.trend_up else sl
    return sum(1 for i in range(lo_i, rng.start)
               if flags[i] and P.confirmed_at(i) <= rng.start)


# ---------------------------------------------------------------- runner
def run_scored(coins, tf, reversal, mode, loader, filters):
    """pa_book.run, with the selection layer applied."""
    rows = []
    kept = seen = 0
    for coin in coins:
        df = loader(coin, tf)
        if df is None or len(df) < 400:
            continue
        o, h, l, c = (df["open"].values, df["high"].values,
                      df["low"].values, df["close"].values)
        sh, sl = P.swing_flags(h, l)
        for rng in P.find_ranges(o, h, l, c):
            seen += 1
            if "location" in filters:
                if not (at_swing_extreme(rng, h, l, sh, sl)
                        or is_decision_point(rng, h, l, c, sh, sl)):
                    continue
            if "decision" in filters and not is_decision_point(
                    rng, h, l, c, sh, sl):
                continue
            if "context" in filters and not reversal:
                if impulse_count(rng, c, sh, sl) > MAX_IMPULSES:
                    continue
            s = P.signals(o, h, l, c, rng, reversal)
            if s is None:
                continue
            eb, side = s
            if "maxpain" in filters:
                mp = max_pain(rng, o, h, l, c, supply=(side < 0))
                if mp is None:
                    continue
                level, beyond = mp
                res = simulate_limit(o, h, l, c, eb, side,
                                     level, beyond, mode)
            else:
                res = P.simulate(o, h, l, c, eb, side, rng, mode)
            if res is None:
                continue
            pnl, bars, r = res
            kept += 1
            rows.append((df.index[eb], coin, side, pnl, bars, r))
    d = pd.DataFrame(rows, columns=["ts", "coin", "side", "pnl",
                                    "bars", "r"]).sort_values("ts")
    return d, seen, kept


def run_htf_ltf(coins, tf, reversal, mode, loader, filters, ltf=1):
    """The book's fourth instruction: find the level on the HIGH timeframe,
    execute on a LOW one (Part 3, Ch 2).

    WHY THIS IS NOT A TUNED PARAMETER. The 12h version measured a
    specific, physical failure: max pain gives a stop around 1.7%, which
    is NARROWER THAN ONE 12h CANDLE on these coins, so 549 of 1021 losses
    came in worse than -1R and 58 of them at -3.3R. You cannot be filled
    at a stop that sits inside a single bar. The book's own answer is to
    take the entry on a lower timeframe, where that same stop is several
    bars wide and can actually be honoured.

    The wait and hold budgets are kept at the SAME WALL-CLOCK length as
    the 12h test (MAX_WAIT * tf hours), so nothing is relaxed -- only the
    resolution of the bars the fill is measured against changes.
    """
    rows = []
    for coin in coins:
        dh = loader(coin, tf)
        dl = loader(coin, ltf)
        if dh is None or dl is None or len(dh) < 400:
            continue
        o, h, l, c = (dh["open"].values, dh["high"].values,
                      dh["low"].values, dh["close"].values)
        lo_, lh, ll, lc = (dl["open"].values, dl["high"].values,
                           dl["low"].values, dl["close"].values)
        sh, sl = P.swing_flags(h, l)
        step = tf // ltf
        for rng in P.find_ranges(o, h, l, c):
            if "decision" in filters and not is_decision_point(
                    rng, h, l, c, sh, sl):
                continue
            if "location" in filters and not (
                    at_swing_extreme(rng, h, l, sh, sl)
                    or is_decision_point(rng, h, l, c, sh, sl)):
                continue
            if "context" in filters and not reversal:
                if impulse_count(rng, c, sh, sl) > MAX_IMPULSES:
                    continue
            s = P.signals(o, h, l, c, rng, reversal)
            if s is None:
                continue
            eb, side = s
            mp = max_pain(rng, o, h, l, c, supply=(side < 0))
            if mp is None:
                continue
            level, beyond = mp
            # the HTF signal bar closes at its right edge; execution can
            # only begin AFTER that, never inside the bar that made it
            t0 = dh.index[eb] + pd.Timedelta(hours=tf)
            j = int(dl.index.searchsorted(t0))
            if j >= len(lc) - 10:
                continue
            res = simulate_limit(lo_, lh, ll, lc, j, side, level, beyond,
                                 mode, wait=P.MAX_WAIT * step,
                                 hold=P.MAX_HOLD * step)
            if res is None:
                continue
            pnl, bars, r = res
            rows.append((dl.index[j], coin, side, pnl, bars, r))
    return pd.DataFrame(rows, columns=["ts", "coin", "side", "pnl",
                                       "bars", "r"]).sort_values("ts")


def selftest() -> int:
    """The filters must REJECT as well as accept, or they do nothing."""
    import event_audit as EA
    import short_risk as SR
    df = EA.load_bars("ETH", 12)
    o, h, l, c = (df["open"].values, df["high"].values,
                  df["low"].values, df["close"].values)
    sh, sl = P.swing_flags(h, l)
    rs = P.find_ranges(o, h, l, c)
    loc = sum(at_swing_extreme(r, h, l, sh, sl) for r in rs)
    dp = sum(is_decision_point(r, h, l, c, sh, sl) for r in rs)
    mp = sum(max_pain(r, o, h, l, c, supply=True) is not None for r in rs)
    imp = [impulse_count(r, c, sh, sl) for r in rs]
    print(f"  ETH 12h: {len(rs)} ranges")
    print(f"    at a swing extreme : {loc:>4}  ({loc/len(rs)*100:.0f}%)")
    print(f"    decision point     : {dp:>4}  ({dp/len(rs)*100:.0f}%)")
    print(f"    max pain definable : {mp:>4}  ({mp/len(rs)*100:.0f}%)")
    print(f"    impulses: median {int(np.median(imp))}, "
          f"<= {MAX_IMPULSES}: {sum(1 for x in imp if x <= MAX_IMPULSES)}")
    assert 0 < loc < len(rs), "location filter accepts all or nothing"
    assert 0 < dp < len(rs), "decision-point filter accepts all or nothing"
    assert min(imp) < max(imp), "impulse count is constant -- not measuring"
    print("  every filter both accepts and rejects: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=12)
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.selftest:
        return selftest()

    import event_audit as EA
    import short_risk as SR
    coins = SR.universe()
    rng_ = np.random.default_rng(zlib.crc32(b"pa_score"))

    stacks = [
        ("none (pa_book)", set()),
        ("+ location", {"location"}),
        ("+ decision point", {"decision"}),
        ("+ dp + max pain", {"decision", "maxpain"}),
        ("+ dp + mp + context", {"decision", "maxpain", "context"}),
    ]
    print(f"\n  Playbook SELECTION layer, {a.tf}h, reversal + fixed 2R")
    print(f"  round trip {P.ROUND_TRIP*100:.3f}%\n")
    print(f"  {'filters':<22}{'n':>6}{'cut':>6}{'win%':>6}{'real':>9}"
          f"{'null':>9}{'edge':>8}{'p':>7}{'H1':>9}{'H2':>9}")
    base_n = None
    for lbl, f in stacks:
        d, seen, kept = run_scored(coins, a.tf, True, "fixed",
                                   EA.load_bars, f)
        if base_n is None:
            base_n = max(len(d), 1)
        if len(d) < 30:
            print(f"  {lbl:<22}{len(d):>6}   too few trades")
            continue
        cut = d.ts.iloc[len(d) // 2]
        h1 = d[d.ts < cut].pnl.mean()
        h2 = d[d.ts >= cut].pnl.mean()
        nl = P.null(coins, a.tf, d, "fixed", EA.load_bars, a.sims, rng_)
        rm = d.pnl.mean()
        p = float((nl >= rm).mean())
        print(f"  {lbl:<22}{len(d):>6}{len(d)/base_n*100:>5.0f}%"
              f"{(d.pnl>0).mean()*100:>5.0f}%{rm*100:>8.3f}%"
              f"{nl.mean()*100:>8.3f}%{(rm-nl.mean())*100:>7.3f}%"
              f"{p:>7.3f}{h1*100:>8.3f}%{h2*100:>8.3f}%")
    print("\n  'cut' is trades kept vs unfiltered. The filters are worth")
    print("  having only if the EDGE rises as the count falls -- a smaller")
    print("  sample at the same edge is just sampling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
