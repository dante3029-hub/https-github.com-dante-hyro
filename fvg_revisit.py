#!/usr/bin/env python3
"""
fvg.py -- Dante's fair-value-gap setups, tested on his terms.

THE RULES AS HE STATED THEM
    "when a fvg is first printed I follow the continuation and then
     later if price re vist red fvg I'll short the re re vist same with
     green"

Two distinct trades, both measured separately:
  FRESH    an FVG prints -> trade the continuation, i.e. in the gap's
           own direction (bullish gap long, bearish gap short)
  REVISIT  price comes back into the gap later -> trade the gap's
           direction again (red gap revisited = short, green = long)

"the re re vist" is ambiguous -- it reads either as emphasis on the
revisit, or as the SECOND return to the same gap. Both are tested
(REVISIT1, REVISIT2) rather than guessing which he meant.

FVG DEFINITION -- the standard three-candle form:
  bullish   low[i] > high[i-2]      gap zone = [high[i-2], low[i]]
  bearish   high[i] < low[i-2]      gap zone = [high[i], low[i-2]]
Detected on the close of bar i; nothing is tradeable before bar i+1, so
no gap is ever acted on in the same bar that completed it.

UNDERSPECIFIED, AND HOW IT IS HANDLED
  stop  -- not stated. Two readings: 1.5 ATR, and the far side of the
           gap (the structural stop most people use on these).
  exit  -- "I sort of exit when I'm happy" from the MACD setup applies
           here too, so a spread of mechanical exits is run and the
           spread itself is the answer.
  size  -- no minimum gap size is imposed, because he did not state one.
           The gap-size distribution is printed so it is visible whether
           the population is dominated by trivial gaps.

Nothing is tuned to fit. Every cell is reported and the best is
corrected for how many were examined.

POWER (claude/verifier-power.md): under ~50 trades this pipeline is
blind; at n=300 it finds anything above +0.17R; at n=1000, +0.07R.
Counts are printed so each verdict can be read against that.

USAGE
    python3 fvg_revisit.py --selftest
    python3 fvg_revisit.py
"""
from __future__ import annotations

import argparse
import logging
import zlib

import numpy as np
import pandas as pd

log = logging.getLogger("fvg")

ATR_LEN = 14
ROUND_TRIP = 0.0017
MAX_HOLD = 60
REVISIT_WINDOW = 200     # bars a gap stays "live" waiting to be revisited
MIN_R_COST = 3.0         # 1R must exceed this many round-trips

# WHY A COST FLOOR IS NOT A TUNED PARAMETER. Gap widths on these coins
# run from 0.03% to 15%. Normalising by R treats a 0.03% stop as one
# unit of risk, which would need ~300x leverage to actually size, and
# the round-trip cost alone is several times that stop. Those rows are
# arithmetic, not trades. Requiring 1R to clear a few round-trips is the
# condition for the trade to exist at all, and it is set from the fee,
# not from what makes the backtest look good.


def ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    a = 2.0 / (n + 1.0)
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = (x[i] - out[i - 1]) * a + out[i - 1]
    return out


def atr(h, l, c, n=ATR_LEN):
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return ema(tr, n)


def find_gaps(h, l):
    """Every three-candle FVG. Returns (bar, side, zlo, zhi).

    side = +1 bullish (green), -1 bearish (red). The gap is complete at
    bar i, so bar i is the earliest bar whose close could have revealed
    it; trading starts at i+1.
    """
    out = []
    for i in range(2, len(h)):
        if l[i] > h[i - 2]:
            out.append((i, 1, float(h[i - 2]), float(l[i])))
        elif h[i] < l[i - 2]:
            out.append((i, -1, float(h[i]), float(l[i - 2])))
    return out


def trade(h, l, c, eb, side, ent, stop, exit_mode):
    """One trade from an explicit entry price.

    `ent` is passed in rather than read off a bar, because a revisit is
    a limit fill at the gap edge, not a market order at a close. A bar
    that touches both stop and target is read as the STOP -- the only
    safe reading from OHLC alone.
    """
    n = len(c)
    if eb >= n or not np.isfinite(ent) or ent <= 0:
        return None
    if side > 0 and stop >= ent:
        return None
    if side < 0 and stop <= ent:
        return None
    rr = abs(stop - ent) / ent
    if rr <= 0 or not np.isfinite(rr):
        return None
    tgt = None
    if exit_mode.endswith("R"):
        tgt = ent * (1 + side * float(exit_mode[:-1]) * rr)
    best = ent
    for i in range(eb, min(eb + MAX_HOLD, n)):
        if side > 0 and l[i] <= stop:
            return -rr - ROUND_TRIP, i - eb, rr
        if side < 0 and h[i] >= stop:
            return -rr - ROUND_TRIP, i - eb, rr
        if tgt is not None:
            if side > 0 and h[i] >= tgt:
                return (tgt / ent - 1) - ROUND_TRIP, i - eb, rr
            if side < 0 and l[i] <= tgt:
                return (1 - tgt / ent) - ROUND_TRIP, i - eb, rr
        if exit_mode == "trail":
            best = max(best, h[i]) if side > 0 else min(best, l[i])
            if side > 0 and c[i] < best * (1 - rr):
                return (c[i] / ent - 1) - ROUND_TRIP, i - eb, rr
            if side < 0 and c[i] > best * (1 + rr):
                return (1 - c[i] / ent) - ROUND_TRIP, i - eb, rr
    last = min(eb + MAX_HOLD, n) - 1
    return side * (c[last] / ent - 1) - ROUND_TRIP, last - eb, rr


def collect(coins, tf, loader, kind, stop_mode, exit_mode):
    """kind: 'fresh' | 'revisit1' | 'revisit2'."""
    rows = []
    gap_sizes = []
    for coin in coins:
        df = loader(coin, tf)
        if df is None or len(df) < 400:
            continue
        h, l, c = df["high"].values, df["low"].values, df["close"].values
        a = atr(h, l, c)
        n = len(c)
        for i, side, zlo, zhi in find_gaps(h, l):
            if i + 1 >= n or np.isnan(a[i]) or a[i] <= 0:
                continue
            gap_sizes.append((zhi - zlo) / c[i])
            if kind == "fresh":
                eb = i + 1
                ent = float(c[i])
                if stop_mode == "atr":
                    stop = ent - side * 1.5 * float(a[i])
                else:
                    # far side of the gap
                    stop = zlo if side > 0 else zhi
                res = trade(h, l, c, eb, side, ent, stop, exit_mode)
                if res is not None and res[2] > MIN_R_COST * ROUND_TRIP:
                    rows.append((df.index[eb], coin, side, *res))
                continue

            # ---- revisit: price returns INTO the gap zone
            want = 1 if kind == "revisit1" else 2
            hits = 0
            eb = None
            # THE NEAR EDGE -- the first price inside the gap that the
            # market actually reaches on its way back.
            #
            # THIS WAS WRONG AND IT MANUFACTURED A HUGE FAKE EDGE. The
            # first version filled a long at zlo (the gap's FLOOR) and a
            # short at zhi (its CEILING) -- in both cases the far side.
            # A bar that merely dipped to the top of a bullish gap was
            # booked as filled at the bottom of it, a price that never
            # traded. Free money the width of the gap, on every trade,
            # which is why the gap-stop cells printed +2.5R per trade
            # and t=49.
            #
            # Price falls INTO a bullish gap from above, so it meets zhi
            # first. It rises INTO a bearish gap from below, so it meets
            # zlo first.
            edge = zhi if side > 0 else zlo
            for j in range(i + 1, min(i + 1 + REVISIT_WINDOW, n)):
                # the bar must actually trade AT the near edge
                reached = (l[j] <= edge) if side > 0 else (h[j] >= edge)
                if not reached:
                    continue
                hits += 1
                if hits == want:
                    eb = j
                    break
            if eb is None:
                continue
            ent = float(edge)
            if stop_mode == "atr":
                stop = ent - side * 1.5 * float(a[eb])
            else:
                stop = (zlo - (zhi - zlo)) if side > 0 else \
                       (zhi + (zhi - zlo))
            res = trade(h, l, c, eb + 1, side, ent, stop, exit_mode)
            if res is not None and res[2] > MIN_R_COST * ROUND_TRIP:
                rows.append((df.index[min(eb + 1, n - 1)], coin, side, *res))
    d = pd.DataFrame(rows, columns=["ts", "coin", "side", "pnl",
                                    "bars", "r"]).sort_values("ts")
    return d, np.array(gap_sizes)


_NULLPOOL: dict = {}
POOL_DRAWS = 2500      # random-entry outcomes precomputed per coin/side


def _pool(coin, tf, loader, side, rmed, exit_mode, rg):
    """Random-entry outcomes for one coin, computed ONCE and reused.

    The first version re-simulated every trade on every one of the 120
    simulations -- roughly six million Python-level trade() calls, which
    is why it never finished. The null only needs the DISTRIBUTION of
    randomly-timed outcomes for a given coin, side, 1R and exit, so that
    distribution is sampled once here and every simulation draws from
    it. Same null, same assumptions, two orders of magnitude cheaper.

    rmed is rounded into the key so that cells with near-identical 1R
    share a pool instead of each building its own.
    """
    key = (coin, tf, side, round(rmed, 4), exit_mode)
    if key in _NULLPOOL:
        return _NULLPOOL[key]
    df = loader(coin, tf)
    if df is None or len(df) < 400:
        _NULLPOOL[key] = np.array([])
        return _NULLPOOL[key]
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(c)
    hi = n - MAX_HOLD - 2
    if hi <= 60:
        _NULLPOOL[key] = np.array([])
        return _NULLPOOL[key]
    outs = []
    for eb in rg.integers(60, hi, size=min(POOL_DRAWS, hi - 60)):
        eb = int(eb)
        ent = float(c[eb])
        stop = ent * (1 - side * rmed)
        res = trade(h, l, c, eb + 1, side, ent, stop, exit_mode)
        if res is not None:
            outs.append(res[0])
    _NULLPOOL[key] = np.array(outs)
    return _NULLPOOL[key]


def null(coins, tf, d, loader, exit_mode, sims, rg):
    """Randomly-timed trades: same coins, same per-coin counts, same side
    mix, same 1R, same exit. Returns the simulated mean per run."""
    if not len(d):
        return np.array([0.0])
    rmed = float(d.r.median())
    # per-coin, per-side counts, so the null matches the real book's mix
    grp = d.groupby(["coin", "side"]).size().to_dict()
    pools = {}
    for (coin, side), k in grp.items():
        p = _pool(coin, tf, loader, int(side), rmed, exit_mode, rg)
        if p.size:
            pools[(coin, side)] = (p, k)
    if not pools:
        return np.array([0.0])
    out = []
    for _ in range(sims):
        acc = []
        for (p, k) in pools.values():
            acc.append(p[rg.integers(0, len(p), k)])
        out.append(float(np.concatenate(acc).mean()))
    return np.array(out)


def top_by_liquidity(coins, loader, k):
    """The k most liquid coins, by median hourly dollar volume.

    Ranked on liquidity, never on outcome, so this does not peek at
    which coins the strategy happened to do well on. It does use the
    whole history to rank, which is a mild full-sample choice -- but it
    is also how anyone picks a universe in practice, and a coin's
    liquidity rank is not what the strategy is being tested on.
    """
    scored = []
    for c in coins:
        df = loader(c, 1)
        if df is None or len(df) < 400 or "volume" not in df:
            continue
        v = (df["close"] * df["volume"]).replace(
            [np.inf, -np.inf], np.nan).dropna()
        if len(v):
            scored.append((float(v.median()), c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:k]]


def selftest() -> int:
    import event_audit as EA
    # synthetic: one unmistakable bullish gap, one bearish, nothing else
    h = np.array([10., 10., 10., 20., 20., 20., 9., 9.])
    l = np.array([9., 9., 9., 11., 19., 19., 8., 8.])
    g = find_gaps(h, l)
    assert (3, 1, 10.0, 11.0) in g, f"bullish gap missed: {g}"
    assert any(s == -1 for _, s, _, _ in g), f"bearish gap missed: {g}"
    print(f"  synthetic: {len(g)} gaps, both directions found")
    # a flat series must produce none
    f = np.ones(50) * 5
    assert not find_gaps(f + 1, f), "gaps found in a flat series"
    print("  flat series produces no gaps: OK")
    df = EA.load_bars("ETH", 4)
    gg = find_gaps(df["high"].values, df["low"].values)
    sz = np.array([(zh - zl) / df["close"].values[i]
                   for i, _, zl, zh in gg])
    print(f"  ETH 4h: {len(gg)} gaps, median size {np.median(sz)*100:.3f}%"
          f", 90th {np.percentile(sz,90)*100:.3f}%")
    assert len(gg) > 50, "implausibly few gaps on real data"
    # entry must never precede the bar that completed the gap
    d, _ = collect(["ETH"], 4, EA.load_bars, "fresh", "atr", "2R")
    assert len(d) > 0 and d.ts.min() > df.index[2], "entry before gap close"
    print("  no entry precedes its gap's completion: OK")

    # EVERY REVISIT FILL MUST BE A PRICE THAT ACTUALLY TRADED.
    # This is the check that was missing: filling at the gap's far edge
    # books a price the market never printed, and it is worth the full
    # width of the gap on every single trade.
    h4, l4 = df["high"].values, df["low"].values
    c4 = df["close"].values
    a4 = atr(h4, l4, c4)
    n4 = len(c4)
    checked = bad = 0
    for i, side, zlo, zhi in find_gaps(h4, l4):
        if i + 1 >= n4 or np.isnan(a4[i]):
            continue
        edge = zhi if side > 0 else zlo
        for j in range(i + 1, min(i + 1 + REVISIT_WINDOW, n4)):
            reached = (l4[j] <= edge) if side > 0 else (h4[j] >= edge)
            if not reached:
                continue
            checked += 1
            # the fill price must sit inside the entry bar's own range
            if not (l4[j] - 1e-9 <= edge <= h4[j] + 1e-9):
                bad += 1
            break
    print(f"  revisit fills checked: {checked}, outside the bar: {bad}")
    assert checked > 50, "too few revisits to validate"
    assert bad == 0, f"{bad} fills at prices that never traded"
    print("  every revisit fill is a traded price: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=150)
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.selftest:
        return selftest()

    import event_audit as EA
    import short_risk as SR
    coins = SR.universe()
    if a.top:
        coins = top_by_liquidity(coins, EA.load_bars, a.top)
        print(f"\n  top {len(coins)} by median hourly $ volume:")
        print("  " + " ".join(coins))
    rg = np.random.default_rng(zlib.crc32(b"fvg"))

    print("\n  Fair value gaps: fresh-print continuation, and revisits")
    print("  green gap -> long, red gap -> short, as stated")
    print("  stop: 1.5 ATR or the far side of the gap\n")
    print(f"  {'tf':>3} {'kind':<10}{'stop':<7}{'exit':<7}{'n':>7}{'1R':>7}"
          f"{'win%':>6}{'R/trade':>9}{'medR':>8}{'nullR':>8}{'edgeR':>8}"
          f"{'t':>7}{'pnl%':>8}")
    cells = []
    shown = False
    for tf in (1, 4):
        for kind in ("fresh", "revisit1", "revisit2"):
            for sm in ("atr", "gap"):
                for xm in ("2R", "3R", "trail"):
                    d, sizes = collect(coins, tf, EA.load_bars, kind,
                                       sm, xm)
                    if not shown and sizes.size:
                        print(f"  [gap sizes {tf}h: median "
                              f"{np.median(sizes)*100:.3f}%, 90th "
                              f"{np.percentile(sizes,90)*100:.3f}%]")
                        shown = True
                    if len(d) < 30:
                        print(f"  {tf:>3} {kind:<10}{sm:<7}{xm:<7}"
                              f"{len(d):>7}   too few")
                        continue
                    R = (d.pnl / d.r).values
                    se = R.std(ddof=1) / np.sqrt(len(R))
                    t = R.mean() / se if se > 0 else 0.0
                    nl = null(coins, tf, d, EA.load_bars, xm, a.sims, rg)
                    rmed = float(d.r.median())
                    nR = float(nl.mean()) / rmed
                    print(f"  {tf:>3} {kind:<10}{sm:<7}{xm:<7}{len(d):>7}"
                          f"{rmed*100:>6.2f}%{(R>0).mean()*100:>5.0f}%"
                          f"{R.mean():>9.3f}{np.median(R):>8.3f}{nR:>8.3f}"
                          f"{R.mean()-nR:>8.3f}{t:>7.2f}"
                          f"{d.pnl.mean()*100:>7.3f}%")
                    cells.append((f"{tf}h {kind}/{sm}/{xm}", len(R),
                                  R.mean(), nR, t))
    if cells:
        N = len(cells)
        E = (np.sqrt(2 * np.log(N))
             - (np.log(np.log(N)) + np.log(4 * np.pi))
             / (2 * np.sqrt(2 * np.log(N))))
        best = max(cells, key=lambda x: x[4])
        print(f"\n  {N} cells examined. E[max t under the null] = {E:.2f}")
        print(f"  best: {best[0]}  n={best[1]}  {best[2]:+.3f}R  "
              f"t={best[4]:.2f}  -> "
              f"{'survives' if best[4] > E else 'does NOT survive'}")
        allR = np.array([c[2] for c in cells])
        print(f"  mean R across cells: {allR.mean():+.3f}  "
              f"(above zero: {(allR > 0).sum()}/{N})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
