#!/usr/bin/env python3
"""
pa_target.py -- the Playbook built the way the book actually says, after
re-reading all 74 pages as IMAGES, not just extracted text.

WHAT THE FIRST TWO ATTEMPTS GOT WRONG. Worth recording, because both
mistakes came from reading the words and skipping the pictures.

  pa_book.py   took every range it found (17,000 trades) and used a
               FIXED 2R target measured from entry.
  pa_score.py  tried to add selectivity, and "fixed" max pain by
               tightening the stop to ~1.8%.

Page 37, verbatim: "My stop goes on the other side of the range, because
I'm betting that the range holds... Even after a clean breakout, price
can come back and wick the lows, WHICH IS WHY I DON'T GO TIGHTER BY
DEFAULT." So the tight stop was not an improvement, it was the opposite
of the instruction. The wide range stop was right the first time.

Page 32, verbatim, and this is the piece that was missing entirely:
  "I don't pick a target by measuring a fixed distance from my entry. I
   aim at the next logical supply or demand zone... The caveat is that I
   want at least 2R out of the trade so the target has to be a minimum
   of that away from my entry. So before I take any trade I ask myself
   one question, which is whether I can get at least 2R out of it before
   the next zone. IF I CAN'T THEN I DON'T TAKE IT."

That is the selection rule. It is not location, it is not max pain, it
is not trend context. It is: measure the distance to the next opposing
zone, divide by the stop distance, and if the answer is under 2 you do
not have a trade. A fixed 2R target can never fail that test, which is
exactly why the first build took 17,000 trades.

Page 37 also sets the thresholds, so they are the book's, not mine:
  continuations  2R-3R   -> MIN_R_CONT = 2.0
  reversals      2.5R-3R -> MIN_R_REV  = 2.5

THE TARGET IS PRIOR STRUCTURE, so there is no lookahead: the next
opposing zone is a confirmed swing that already exists on the chart at
the moment of entry. Nothing here reads a bar the market has not printed.

THE BAR IT HAS TO CLEAR is unchanged: beat a randomly timed trade of the
same side with the same stop and the same exit. On this universe a
random short earns money from alt-perp drift alone.

USAGE
    python3 pa_target.py --tf 12
    python3 pa_target.py --selftest
"""
from __future__ import annotations

import argparse
import logging
import zlib

import numpy as np
import pandas as pd

import pa_book as P

log = logging.getLogger("pa_target")

MIN_R_CONT = 2.0      # p.37 "2R-3R is a good safe bet for continuations"
MIN_R_REV = 2.5       # p.37 "For reversals I look more for 2.5R to 3R"
ZONE_LOOKBACK = 60    # bars back to find the next opposing zone


def next_zone(zones, at: int, side: int, entry: float, rng: P.Rng):
    """The next logical supply or demand zone (p.32).

    FIRST VERSION WAS WRONG and the symptom gave it away: median
    available-R came out at 0.15, i.e. the "next zone" was seven times
    closer than the stop on a typical trade. It was picking the nearest
    confirmed swing of any size, and there is always some minor swing a
    few bars back sitting right next to price.

    Page 14 defines what the target actually is: a price action zone is
    "the space between a demand zone and a supply zone", and price
    crossing into one "is going to travel to the other side of it". The
    trade is the crossing. So the target zone must sit BEYOND THE RANGE
    on the far side from the stop -- a swing inside the range is not the
    other side of anything, it is the chop the range is made of.

    SECOND VERSION WAS ALSO WRONG, for a subtler reason. Restricting
    candidates to swings beyond the range changed nothing: median
    available-R stayed at 0.15. The real fault is that a 2-bar fractal
    is not a supply or demand zone. The book never marks every wiggle --
    it marks the four formations (RBR, DBD, DBR, RBD), and p.48 says
    "if you look at any chart I post, this is all you'll see marked. I
    only mark the extremes."

    So the zones here are the boundaries of OTHER RANGES the same
    detector already found. That introduces no new threshold: a zone is
    whatever the book's own range rule calls a zone, and the target is
    the nearest one beyond this range on the far side from the stop.

    A zone counts only once its own range was confirmed (z.start <= at),
    so nothing here reads a bar the market has not printed. Returns None
    when there is no zone out there, in which case the book's question
    has no answer and the trade is skipped.
    """
    if side < 0:
        bound = min(entry, rng.lo)
        cands = [z.lo for z in zones if z.start <= at and z.lo < bound]
        return max(cands) if cands else None     # nearest one below
    bound = max(entry, rng.hi)
    cands = [z.hi for z in zones if z.start <= at and z.hi > bound]
    return min(cands) if cands else None          # nearest one above


def simulate_to_zone(o, h, l, c, eb, side, rng: P.Rng, target: float):
    """Enter at the open after the signal, stop the far side of the RANGE
    (p.37), exit at the next opposing zone (p.32).

    A bar that touches both stop and target is read as the stop -- the
    conservative assumption when only OHLC is known.
    """
    n = len(c)
    if eb >= n:
        return None
    ent = o[eb]
    stop = rng.lo if side > 0 else rng.hi
    r = abs(ent - stop) / ent
    if r <= 0 or not np.isfinite(r):
        return None
    for i in range(eb, min(eb + P.MAX_HOLD, n)):
        if side > 0 and l[i] <= stop:
            return (stop / ent - 1) - P.ROUND_TRIP, i - eb, r
        if side < 0 and h[i] >= stop:
            return (1 - stop / ent) - P.ROUND_TRIP, i - eb, r
        if side > 0 and h[i] >= target:
            return (target / ent - 1) - P.ROUND_TRIP, i - eb, r
        if side < 0 and l[i] <= target:
            return (1 - target / ent) - P.ROUND_TRIP, i - eb, r
    last = min(eb + P.MAX_HOLD, n) - 1
    return (side * (c[last] / ent - 1)) - P.ROUND_TRIP, last - eb, r


def sparse(ranges):
    """One live range at a time -- the way a person actually trades.

    THIS IS THE REAL PROBLEM WITH ALL THREE BUILDS SO FAR, and it took
    three separate symptoms to see it:
      - 93% of ranges sat "at a swing extreme" (location filter is a
        no-op)
      - the "next zone" was 0.06-0.15R away (target rule degenerates)
      - 17,000 trades where the author takes two or three a week

    One cause underneath all of them: find_ranges emits 556 overlapping,
    nested ranges on a single chart. Every rule the book gives -- trade
    only the extremes, target the next zone, require 2R before it --
    assumes zones are SPARSE. Make them dense and each rule degenerates
    into noise, which is exactly what was measured.

    No new threshold is introduced here. The rule is causal and
    structural: once a range is confirmed it is the live one, and any
    range confirmed while its price band is still in play is a redraw of
    the same level, not a second setup. That is the book's "clean" and
    "obvious" turned into something a machine can apply without a number
    being chosen to make it work.
    """
    out = []
    live = None
    for r in sorted(ranges, key=lambda z: z.start):
        if live is not None:
            overlaps = not (r.lo >= live.hi or r.hi <= live.lo)
            if overlaps:
                continue              # same level redrawn -- skip
        out.append(r)
        live = r
    return out


def run(coins, tf, reversal, loader, gate=True, thin=True):
    """gate=True applies the book's 2R-to-next-zone test; gate=False runs
    the same trades without it, so the filter can be measured on its own.
    """
    rows = []
    seen = 0
    for coin in coins:
        df = loader(coin, tf)
        if df is None or len(df) < 400:
            continue
        o, h, l, c = (df["open"].values, df["high"].values,
                      df["low"].values, df["close"].values)
        sh, sl = P.swing_flags(h, l)
        need = MIN_R_REV if reversal else MIN_R_CONT
        ranges = P.find_ranges(o, h, l, c)
        if thin:
            ranges = sparse(ranges)
        for rng in ranges:
            s = P.signals(o, h, l, c, rng, reversal)
            if s is None:
                continue
            eb, side = s
            if eb >= len(c):
                continue
            seen += 1
            ent = o[eb]
            stop = rng.lo if side > 0 else rng.hi
            r = abs(ent - stop) / ent
            if r <= 0 or not np.isfinite(r):
                continue
            tz = next_zone(ranges, eb, side, ent, rng)
            if tz is None:
                continue
            avail = abs(tz - ent) / ent / r          # R to the next zone
            if gate and avail < need:
                continue                              # "I don't take it"
            tgt = tz if gate else ent * (1 + side * need * r)
            res = simulate_to_zone(o, h, l, c, eb, side, rng, tgt)
            if res is None:
                continue
            pnl, bars, rr = res
            rows.append((df.index[eb], coin, side, pnl, bars, rr, avail))
    d = pd.DataFrame(rows, columns=["ts", "coin", "side", "pnl", "bars",
                                    "r", "availR"]).sort_values("ts")
    return d, seen


def selftest() -> int:
    import event_audit as EA
    df = EA.load_bars("ETH", 12)
    h, l = df["high"].values, df["low"].values
    sh, sl = P.swing_flags(h, l)
    n = len(h)
    mids = [(h[i] + l[i]) / 2 for i in range(200, n - 1, 37)]
    rngs = [P.Rng(lo=m * 0.98, hi=m * 1.02, start=i, trend_up=True)
            for m, i in zip(mids, range(200, n - 1, 37))]
    zs = P.find_ranges(df["open"].values, h, l, df["close"].values)
    got = [next_zone(zs, i, -1, m, rr)
           for i, m, rr in zip(range(200, n - 1, 37), mids, rngs)]
    ok = [g for g in got if g is not None]
    print(f"  ETH 12h: {len(ok)}/{len(got)} bars have a demand zone below")
    assert ok, "next_zone never finds structure -- it is not measuring"
    for g, m in zip(got, mids):
        assert g is None or g < m * 0.98, "zone not beyond the range low"
    print("  next_zone: finds, skips, and never sits on the wrong side: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=0)
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.selftest:
        return selftest()

    import event_audit as EA
    import short_risk as SR
    coins = SR.universe()
    rg = np.random.default_rng(zlib.crc32(b"pa_target"))
    tfs = [a.tf] if a.tf else [4, 6, 12]

    print("\n  Playbook with the book's OWN target rule (p.32 + p.37)")
    print("  stop = far side of range | target = next opposing zone")
    print(f"  gate: continuations >={MIN_R_CONT}R, reversals "
          f">={MIN_R_REV}R to the next zone, else no trade\n")
    print(f"  {'setup':<24}{'n':>6}{'kept':>6}{'1R':>7}{'win%':>6}"
          f"{'R/trade':>9}{'nullR':>8}{'edgeR':>8}{'p':>7}{'H1':>8}{'H2':>8}")
    for tf in tfs:
        for rev in (False, True):
            d, seen = run(coins, tf, rev, EA.load_bars, gate=True)
            lbl = f"{tf}h {'reversal' if rev else 'continuation'}"
            if len(d) < 30:
                print(f"  {lbl:<24}{len(d):>6}   too few trades")
                continue
            R = d.pnl / d.r
            cut = d.ts.iloc[len(d) // 2]
            h1 = float((d[d.ts < cut].pnl / d[d.ts < cut].r).mean())
            h2 = float((d[d.ts >= cut].pnl / d[d.ts >= cut].r).mean())
            nl = P.null(coins, tf, d, "fixed", EA.load_bars, a.sims, rg)
            rmed = float(d.r.median())
            nlR = nl / rmed
            p = float((nl >= d.pnl.mean()).mean())
            print(f"  {lbl:<24}{len(d):>6}{len(d)/max(seen,1)*100:>5.0f}%"
                  f"{rmed*100:>6.1f}%{(R>0).mean()*100:>5.0f}%"
                  f"{R.mean():>9.3f}{nlR.mean():>8.3f}"
                  f"{R.mean()-nlR.mean():>8.3f}{p:>7.3f}{h1:>8.3f}{h2:>8.3f}")
    print("\n  'kept' is the share of signals that passed the 2R-to-zone")
    print("  gate. That gate is the book's selectivity; everything else")
    print("  it says about choosing levels did not survive being coded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
