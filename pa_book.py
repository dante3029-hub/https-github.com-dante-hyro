#!/usr/bin/env python3
"""
pa_book.py -- TraderMatt's Price Action Playbook, implemented and tested.

THE METHOD, as the book states it (Part 2, Ch 1-3; Part 3, Ch 1):
  1. A trend puts in a high (or low).
  2. A retrace puts in the opposite extreme. Range = [retrace low, trend
     high] once the trend high is RETESTED.
  3. Price builds inside the range.
  4. FIRST break out of the range is a "signal of intent", NOT a trigger.
  5. Price comes back into the range or retests it.
  6. SECOND break, in the trend's direction, on a MOMENTUM candle, is the
     entry -- at the next bar's open, never on an open candle.
  Reversals (break against the trend) need a THIRD break.
  Stop goes the other side of the range. Target 2R minimum.

EVERY THRESHOLD IS FIXED FROM THE BOOK'S OWN TEXT, NOT TUNED. That is
the entire point. This project has run four parameter sweeps -- 30
Supertrend cells, 18 candlesticks, 8 pattern formations, 30
timeframe/exit cells -- and the only one that produced anything
out-of-sample was the one where thresholds were fixed in advance. A
sweep over `swing_k`, `momentum_mult` and `max_wait` would find a
winning cell here too, and it would mean nothing.

  swing_k       = 2     standard fractal; the book draws swings by eye
  momentum_mult = 1.5   "a candle bigger than all the others around it"
  min_r         = 2.0   "I want at least 2R out of the trade"
  max_wait      = 50    bars allowed between first break and entry.
                        The book says "wait", with no number. A cap is
                        needed or a range stays live forever, so this is
                        an arbitrary but PRE-REGISTERED choice.
  max_hold      = 200   bars before abandoning an open trade.

THE BAR IT HAS TO CLEAR is the same as everything else here: beat a
RANDOMLY TIMED trade of the same direction with the SAME exit logic,
and be positive in both independent time halves. On this universe a
random 10-bar short earns +1.2% gross because alt perps drift down, so
"it made money" is not evidence of anything.

HONEST MISMATCH, stated up front: the book is written for discretionary
M1 Bitcoin trading in one session. This tests it mechanically across 63
alt perps on 4h/6h/12h bars. A null result would be evidence against
the mechanical version on this data, not against Matt's trading.

USAGE
    python3 pa_book.py --tf 12 --mode both
    python3 pa_book.py --tf 6 --mode continuation --exit trail
    python3 pa_book.py --selftest
"""
from __future__ import annotations

import argparse
import logging
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("pa_book")

SWING_K = 2
MOMENTUM_MULT = 1.5
MIN_R = 2.0
MAX_WAIT = 50
MAX_HOLD = 200
FEE = 0.00085            # per side, same as every other test here
ROUND_TRIP = 2 * FEE


# ----------------------------------------------------------------- swings
def swing_flags(h: np.ndarray, l: np.ndarray, k: int = SWING_K):
    """(is_swing_high, is_swing_low) fractals.

    A swing high at i needs k bars either side with LOWER highs. That
    means it is only CONFIRMED k bars later -- the caller must never use
    a swing before index i+k, or it is reading the future. Enforced by
    `confirmed_at` below rather than left to discipline.
    """
    n = len(h)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        wh = h[i - k:i + k + 1]
        wl = l[i - k:i + k + 1]
        if h[i] == wh.max() and (wh.argmax() == k):
            sh[i] = True
        if l[i] == wl.min() and (wl.argmin() == k):
            sl[i] = True
    return sh, sl


def confirmed_at(i: int, k: int = SWING_K) -> int:
    """Bar at which a swing at i becomes knowable."""
    return i + k


# ------------------------------------------------------------------ range
@dataclass
class Rng:
    lo: float
    hi: float
    start: int          # bar the range was confirmed (knowable from here)
    trend_up: bool      # the trend INTO the range


def find_ranges(o, h, l, c, k=SWING_K):
    """Yield Rng objects, built the book's way (Part 1, Ch 4).

    After a move up: trend high -> retrace low -> the HIGH IS RETESTED.
    The retest is what confirms the range, so the range is not tradeable
    until the bar where price returns to the trend high.
    """
    n = len(c)
    sh, sl = swing_flags(h, l, k)
    highs = [i for i in range(n) if sh[i]]
    lows = [i for i in range(n) if sl[i]]
    out = []

    # --- after a move up: high, then retrace low, then retest of the high
    for hi_i in highs:
        nxt = [j for j in lows if j > hi_i]
        if not nxt:
            continue
        lo_i = nxt[0]
        ready = confirmed_at(lo_i, k)
        if ready >= n:
            continue
        # retest: price trades back up to the trend high
        rt = None
        for j in range(ready, min(ready + MAX_WAIT, n)):
            if h[j] >= h[hi_i]:
                rt = j
                break
        if rt is None:
            continue
        out.append(Rng(lo=l[lo_i], hi=h[hi_i], start=rt, trend_up=True))

    # --- after a move down: low, then retrace high, then retest of the low
    for lo_i in lows:
        nxt = [j for j in highs if j > lo_i]
        if not nxt:
            continue
        hi_i = nxt[0]
        ready = confirmed_at(hi_i, k)
        if ready >= n:
            continue
        rt = None
        for j in range(ready, min(ready + MAX_WAIT, n)):
            if l[j] <= l[lo_i]:
                rt = j
                break
        if rt is None:
            continue
        out.append(Rng(lo=l[lo_i], hi=h[hi_i], start=rt, trend_up=False))

    out.sort(key=lambda r: r.start)
    return out


# -------------------------------------------------------------- the entry
def momentum(c, o, i, lo_i, hi_i, mult=MOMENTUM_MULT) -> bool:
    """Is bar i far bigger than the candles inside the range?

    The book: "a candle that's bigger than all the others around it...
    volume has stepped in". Measured on BODY, because the book cares
    about the CLOSE ("the close is the one I care about most, because it
    tells you who won") -- a big wick that closes back inside is exactly
    the fakeout it warns about.
    """
    if hi_i <= lo_i:
        return False
    bodies = np.abs(c[lo_i:hi_i] - o[lo_i:hi_i])
    bodies = bodies[bodies > 0]
    if bodies.size < 3:
        return False
    return abs(c[i] - o[i]) > mult * float(bodies.mean())


def signals(o, h, l, c, rng: Rng, reversal: bool):
    """The double (or triple) breakout. Returns (entry_bar, side) or None.

    Continuation = break WITH the trend into the range.
    Reversal     = break AGAINST it, and needs one more break.
    """
    n = len(c)
    want = (1 if rng.trend_up else -1) * (-1 if reversal else 1)
    need = 3 if reversal else 2

    breaks = 0
    back_inside = True
    for i in range(rng.start, min(rng.start + MAX_WAIT, n - 1)):
        out_up = c[i] > rng.hi
        out_dn = c[i] < rng.lo
        inside = not (out_up or out_dn)

        if inside:
            back_inside = True
            continue
        if not back_inside:
            continue                       # already counted this excursion

        side = 1 if out_up else -1
        back_inside = False
        breaks += 1

        if breaks < need:
            continue
        if side != want:
            # broke the wrong way on the deciding break: not our trade
            return None
        if not momentum(c, o, i, rng.start, i):
            breaks -= 1                    # weak break doesn't count
            back_inside = True
            continue
        return i + 1, side                 # enter at the NEXT bar's open
    return None


# --------------------------------------------------------------- the exit
def simulate(o, h, l, c, eb, side, rng: Rng, mode: str):
    """Run the trade. mode: 'fixed' = 2R target, 'trail' = the book's
    structure trail (Part 2, Ch 6).

    THE TRAIL, as written: "when a third higher high is made, move the
    stop under every second higher low... once it's past 2R, move it
    under every higher low." Tracked forward bar by bar; a swing is only
    used once CONFIRMED, so the trail never sees a low before the market
    did.
    """
    n = len(c)
    if eb >= n:
        return None
    ent = o[eb]
    stop = rng.lo if side > 0 else rng.hi
    r = abs(ent - stop) / ent
    if r <= 0 or not np.isfinite(r):
        return None
    tgt = ent * (1 + side * MIN_R * r)

    hh = []          # confirmed higher highs (lower lows for a short)
    hl = []          # confirmed higher lows  (lower highs for a short)
    best = 0.0
    for i in range(eb, min(eb + MAX_HOLD, n)):
        # stop first: a bar that hits both is assumed to hit the stop,
        # the conservative read when only OHLC is known.
        if side > 0 and l[i] <= stop:
            return (stop / ent - 1) - ROUND_TRIP, i - eb, r
        if side < 0 and h[i] >= stop:
            return (1 - stop / ent) - ROUND_TRIP, i - eb, r
        if mode == "fixed":
            if side > 0 and h[i] >= tgt:
                return (tgt / ent - 1) - ROUND_TRIP, i - eb, r
            if side < 0 and l[i] <= tgt:
                return (1 - tgt / ent) - ROUND_TRIP, i - eb, r
        else:
            prog = side * (c[i] / ent - 1) / r
            best = max(best, prog)
            j = i - SWING_K                      # newly confirmed swing
            if j > eb:
                w = slice(j - SWING_K, j + SWING_K + 1)
                if side > 0:
                    if h[j] == h[w].max():
                        hh.append(h[j])
                    if l[j] == l[w].min():
                        hl.append(l[j])
                    if best >= MIN_R and hl:
                        stop = max(stop, hl[-1])
                    elif len(hh) >= 3 and len(hl) >= 2:
                        stop = max(stop, hl[-2])
                else:
                    if l[j] == l[w].min():
                        hh.append(l[j])
                    if h[j] == h[w].max():
                        hl.append(h[j])
                    if best >= MIN_R and hl:
                        stop = min(stop, hl[-1])
                    elif len(hh) >= 3 and len(hl) >= 2:
                        stop = min(stop, hl[-2])
    i = min(eb + MAX_HOLD, n - 1)
    return side * (c[i] / ent - 1) - ROUND_TRIP, i - eb, r


# ---------------------------------------------------------------- harness
def run(coins, tf, reversal, mode, loader):
    rows = []
    for coin in coins:
        df = loader(coin, tf)
        if df is None or len(df) < 400:
            continue
        o, h, l, c = (df["open"].values, df["high"].values,
                      df["low"].values, df["close"].values)
        for rng in find_ranges(o, h, l, c):
            s = signals(o, h, l, c, rng, reversal)
            if s is None:
                continue
            eb, side = s
            res = simulate(o, h, l, c, eb, side, rng, mode)
            if res is None:
                continue
            pnl, bars, r = res
            rows.append((df.index[eb], coin, side, pnl, bars, r))
    return pd.DataFrame(rows, columns=["ts", "coin", "side", "pnl",
                                       "bars", "r"]).sort_values("ts")


def null(coins, tf, real, mode, loader, sims, rng_):
    """Random entry bars, SAME side mix, SAME range-derived stop and the
    same exit. Without this the alt-perp downward drift alone makes any
    short-heavy rule look profitable."""
    per = {}
    for coin in coins:
        df = loader(coin, tf)
        if df is None or len(df) < 400:
            continue
        per[coin] = df
    counts = real.groupby("coin").size().to_dict()
    sides = real.side.values
    if not len(sides):
        return np.array([0.0])
    out = []
    for _ in range(sims):
        acc = []
        for coin, k in counts.items():
            df = per.get(coin)
            if df is None:
                continue
            o, h, l, c = (df["open"].values, df["high"].values,
                          df["low"].values, df["close"].values)
            n = len(c)
            for _i in range(k):
                eb = int(rng_.integers(60, n - 60))
                sd = int(rng_.choice(sides))
                # a synthetic range of the same typical width, so the
                # stop distance matches the real trades' 1R
                width = float(np.median(real.r.values)) * c[eb]
                rr = Rng(lo=c[eb] - width, hi=c[eb] + width,
                         start=eb, trend_up=sd > 0)
                res = simulate(o, h, l, c, eb, sd, rr, mode)
                if res is not None:
                    acc.append(res[0])
        if acc:
            out.append(float(np.mean(acc)))
    return np.array(out) if out else np.array([0.0])


def selftest() -> int:
    """Synthetic bars with a hand-built double breakout."""
    # FLAT REPEATED BARS DO NOT WORK as a fixture, and the first version
    # of this test was built from them. Identical lows mean argmin never
    # lands on the centre bar, so swing_flags finds nothing and the test
    # failed for a reason that has nothing to do with the method. On real
    # ETH 12h data ties block 0.5% of lows and 0.0% of highs, so the
    # strictness is fine -- the fixture was the problem. Every bar below
    # therefore has distinct highs and lows.
    o = []; h = []; l = []; c = []

    def bar(op, hi, lo, cl):
        o.append(op); h.append(hi); l.append(lo); c.append(cl)

    bar(95, 97, 94, 96)
    bar(96, 99, 95, 98)
    bar(98, 101, 97, 100)
    bar(100, 103, 99, 102)
    bar(102, 110, 101, 109)           # bar 4: trend high 110
    bar(109, 108, 104, 105)
    bar(105, 106, 100, 101)           # bar 6: retrace low 100
    bar(101, 107, 102, 106)
    bar(106, 110, 105, 109)           # bar 8: retests 110 -> range live
    bar(109, 108.5, 103.5, 104)       # builds inside, distinct values
    bar(104, 107.5, 103, 106)
    bar(106, 108.2, 104.2, 105)
    bar(105, 106.8, 103.2, 104.5)
    bar(104.5, 112, 104, 111)         # bar 13: FIRST break (close > 110)
    bar(111, 111.5, 104.8, 105.5)     # bar 14: back inside
    bar(105.5, 107.2, 103.8, 106.2)
    bar(106.2, 108.4, 104.6, 105.2)
    bar(105.2, 107.9, 104.1, 106.8)
    bar(106.8, 125, 106, 124)         # bar 18: SECOND break, big body
    for g in range(20):               # runs, distinct each bar
        bar(124 + g, 140 + g, 123 + g, 138 + g)
    o, h, l, c = (np.array(x, float) for x in (o, h, l, c))

    rs = find_ranges(o, h, l, c)
    assert rs, "no range built from a textbook setup"
    r0 = [r for r in rs if r.trend_up and abs(r.hi - 110) < 1e-9]
    assert r0, f"range high should be the trend high 110, got {[x.hi for x in rs]}"
    s = signals(o, h, l, c, r0[0], reversal=False)
    assert s is not None, "double breakout not detected"
    eb, side = s
    assert side == 1, "should be a long continuation"
    assert c[eb - 1] > 110, "entry must follow a close above the range"
    res = simulate(o, h, l, c, eb, side, r0[0], "fixed")
    assert res is not None and res[0] > 0, f"textbook winner came out {res}"
    print(f"  range {r0[0].lo:.0f}-{r0[0].hi:.0f}, entry bar {eb}, "
          f"side {side:+d}, pnl {res[0]*100:+.2f}%")

    # the FIRST break alone must not trigger
    s2 = signals(o[:16], h[:16], l[:16], c[:16], r0[0], reversal=False)
    assert s2 is None, "fired on the first break -- that is the whole point"
    print("  first break alone does NOT trigger: OK")
    print("  selftest passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=12)
    ap.add_argument("--mode", default="both",
                    choices=["continuation", "reversal", "both"])
    ap.add_argument("--exit", dest="ex", default="both",
                    choices=["fixed", "trail", "both"])
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.selftest:
        return selftest()

    import event_audit as EA
    import short_risk as SR
    loader = EA.load_bars
    coins = SR.universe()
    rng_ = np.random.default_rng(zlib.crc32(b"pa_book"))

    kinds = ([("continuation", False), ("reversal", True)]
             if a.mode == "both" else
             [(a.mode, a.mode == "reversal")])
    exits = ["fixed", "trail"] if a.ex == "both" else [a.ex]

    print(f"\n  TraderMatt Playbook, {a.tf}h bars, {len(coins)} coins."
          f"  round trip {ROUND_TRIP*100:.3f}%")
    print(f"  thresholds fixed from the book: swing_k={SWING_K}, "
          f"momentum={MOMENTUM_MULT}x, min_r={MIN_R}, wait<={MAX_WAIT}\n")
    print(f"  {'pattern':<14}{'exit':<7}{'n':>6}{'win%':>6}{'real':>9}"
          f"{'null':>9}{'edge':>8}{'p':>7}{'H1':>9}{'H2':>9}{'ok':>4}")
    for kname, krev in kinds:
        for ex in exits:
            d = run(coins, a.tf, krev, ex, loader)
            if len(d) < 40:
                print(f"  {kname:<14}{ex:<7}{len(d):>6}   too few trades")
                continue
            cut = d.ts.iloc[len(d) // 2]
            h1 = d[d.ts < cut].pnl.mean()
            h2 = d[d.ts >= cut].pnl.mean()
            nl = null(coins, a.tf, d, ex, loader, a.sims, rng_)
            rm = d.pnl.mean()
            p = float((nl >= rm).mean())
            ok = "YES" if (h1 > 0 and h2 > 0 and p < 0.05) else ""
            print(f"  {kname:<14}{ex:<7}{len(d):>6}"
                  f"{(d.pnl>0).mean()*100:>5.0f}%{rm*100:>8.3f}%"
                  f"{nl.mean()*100:>8.3f}%{(rm-nl.mean())*100:>7.3f}%"
                  f"{p:>7.3f}{h1*100:>8.3f}%{h2*100:>8.3f}%{ok:>4}")
    print("\n  'real' and 'null' are NET of fees. The null is a randomly")
    print("  timed trade, same side mix, same stop distance, same exit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
