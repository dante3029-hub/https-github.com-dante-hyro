#!/usr/bin/env python3
"""
tf_confirm.py -- is the 12-16h hump in V29's returns an effect, or drift?

WHERE THIS CAME FROM. tf_sweep.py swept V29 over 11 timeframes and gross
return per trade rose smoothly from +0.048% at 1h to +0.796% at 14h, then
collapsed to -1.864% at 24h. The peak and its two neighbours (12h +0.634%,
14h +0.796%, 16h +0.532%) all clear the 0.170% market-order round trip, so
on the face of it V29 was being run roughly 14x too fast. That survived a
real backtest bug being fixed (the Supertrend trail was gated on TP1, so
unresolved trades ran to the end of the data; 14h moved 0.806% -> 0.796%).

THREE REASONS THAT IS NOT YET A RESULT, and this file tests all three.

  1. DRIFT. Gross per trade scaling roughly LINEARLY with bar size is the
     signature of simply holding longer, not of a better signal. 1h to 14h
     is 14x the bar length and 0.048% -> 0.796% is 16x the return. Over a
     sample where the universe rose 1414%, any long-biased rule that holds
     5-6 bars collects more drift per trade at 14h than at 1h by
     construction, and that is arithmetic, not alpha.

     THE TEST: random-timing null at the peak timeframe, run twice.
       null A  random entry bar, random side
       null B  random entry bar, SAME side as the real trade
     Null B is the one that matters. It holds the long/short mix and the
     hold length fixed and randomises ONLY when you enter. If null B
     reproduces the real number, the Supertrend/EMA/ADX entry contributes
     nothing and the hump is the hold length.

     Plus the LONG/SHORT split. If longs earn it and shorts earn nothing,
     the answer is drift whatever the null says.

  2. THE MAXIMUM OF 11 DRAWS. I swept 11 timeframes and am reading the
     largest. Even on pure noise the max of 11 is biased upward, and the
     fact that the curve is smooth does not fix that -- adjacent
     timeframes are resamples of the SAME bars, so their errors are
     strongly correlated and a smooth hump is exactly what correlated
     noise looks like. A hump is not independent confirmation of its peak.

     THE TEST: split the sample in half by time and sweep each half
     separately. If both halves peak in the 12-16h band, the location is a
     property of the market. If one peaks at 6h and the other at 20h, the
     hump is the average of two unrelated noise curves and there is
     nothing there.

  3. IT STILL LOSES TO BUY-AND-HOLD. 14h returns +1333% against +1414% for
     holding the same coins. That is reported again here and not buried.
     It matters less than it looks for the stated purpose -- buy-and-hold
     is not available on an evaluation with a 10% floor, and 2,839 trades
     of market-neutral-ish exposure is a different risk object from being
     long 63 alts through a 70% drawdown -- but it does mean this is not
     an alpha machine, and anyone claiming otherwise should have to beat
     the simplest possible benchmark first.

WHAT WOULD MAKE ME BELIEVE IT, written down before the run:
  - null B p <= 0.05 at the peak, AND
  - both time halves peaking inside 10-16h, AND
  - shorts not materially negative.
Two of three is interesting and not actionable. One of three is noise.

----------------------------------------------------------------------
WHAT ACTUALLY HAPPENED (63 coins, 2023-02 -> 2026-09, 14h unless noted)

The pre-registered criteria were the wrong criteria, and not in a way
that flattered the strategy. They were all stated at the level of the
WHOLE BOOK, and the whole book is two strategies with opposite
properties averaged together:

    long    1,506 trades   +0.025% gross   random entries: +0.092%
    short   1,243 trades   +1.058% gross   random entries: -0.188%

The long book is not a weak signal. It earns LESS than opening the same
number of trades at random bars on the same coins, and it pays 0.170%
a round trip to do it. The short book earns six times the cost bar, and
a randomly timed short in this universe LOSES money -- so the short
result is not downward drift either, even though the median coin in the
sample fell 50.8%.

Pooling them hid this completely. The whole-book halves sweep read
+0.007% at 14h in the second half and I was one step from writing V29
off on that number. It is flat because a dead long book of equal size
cancels a live short book. Testing the average of a dead strategy and a
live one is a way to fail to find something that is there, and it is
the mistake this file was built to catch in the data and then made in
its own design.

Short book alone, gross per trade, both halves:
         full    1st half   2nd half
    10h  0.860%    0.714%     0.783%
    12h  1.251%    0.857%     1.331%
    14h  1.058%    0.786%     0.951%
Against its own timing null at 12h: z=2.00, p=0.0365 full sample, with
the sign agreeing in both halves (z=1.37, z=1.91).

That nominal p does not survive the 33 cells it took to find it
(x7 for correlated speeds: 0.26). What does not depend on a p-value is
the same sign at three adjacent speeds in two disjoint time periods.
That is replication, not significance, and it is the first result in
this project that has shown both.

ALSO SETTLED: the "buy & hold +1414%" row every V29 comparison was
judged against is a SUM over 63 coins dominated by ZEC at +3231%.
Excluding the top three names, holding this universe returns -1386%.
"V29 does not beat buy-and-hold" was never the indictment it sounded
like, and it should not have been reported eleven times as though it
were.

A NOTE ON THE DATA: the account owner's taker_data and this snapshot
are not the same universe (his b&h sum reads 1414%, this one 2650%, and
his is topped up ~3 weeks further). The 14h gross differs accordingly,
0.796% there against 0.492% here. The SHAPE and the PEAK LOCATION
replicate across the two snapshots; the level does not. Treat the level
as unknown to within a factor of ~1.6.
----------------------------------------------------------------------

USAGE
    python3 -u tf_confirm.py                  # all three tests, 14h peak
    python3 -u tf_confirm.py --tf 12
    python3 -u tf_confirm.py --sims 400
    python3 -u tf_confirm.py --skip-halves    # just the null + split
"""
from __future__ import annotations

import argparse
import glob
import os
import zlib

import numpy as np

import v29_test as V

BAR_MKT = 0.170      # market-order round trip, % of notional
HALVES_GRID = (4, 6, 8, 10, 12, 14, 16, 20)
# 1h and 2h are omitted from the halves sweep only. They are 40,000-bar
# pure-Python indicator passes per coin and both are already far under the
# maker bar on the full sample; spending twenty minutes to confirm a dead
# region in each half buys nothing.


def log(m=""):
    print(m, flush=True)


def universe():
    """Every coin in taker_data except BTC, which is the regime input."""
    out = []
    for f in sorted(glob.glob(f"{V.PERP}/*_1h.csv")):
        c = os.path.basename(f).replace("_1h.csv", "")
        if c != "BTC":
            out.append(c)
            V.COIN_CONFIGS.setdefault(c, (21, 2.0))
    return out


def gross(trades):
    """Weighted gross % per trade. v29_test returns net of 2x FEE."""
    if not trades:
        return float("nan"), 0
    r = np.array([t[0] for t in trades]) + 2 * V.FEE
    w = np.array([t[1] for t in trades])
    if w.sum() <= 0:
        return float("nan"), 0
    return float(np.sum(r * w) / np.sum(w)) * 100.0, len(trades)


def collect(coins, tf, lo=None, hi=None):
    """Run V29 over `coins` at `tf`, optionally on a bar slice. Returns
    {coin: trades} plus the frames and BTC series, so the null can reuse
    exactly the same objects."""
    V.TF = tf
    per, frames, btcs = {}, {}, {}
    for c in coins:
        df = V._load(f"{V.PERP}/{c}_1h.csv")
        if df is None:
            continue
        if tf > 1:
            df = V._resample(df, tf)
        if lo is not None:
            df = df.iloc[lo:hi]
        if len(df) < 300:
            continue
        btc = V.btc_macd_dir(df.index)
        tr = V.run_coin(c, df, btc=btc)
        if tr:
            per[c], frames[c], btcs[c] = tr, df, btc
    return per, frames, btcs


# ---------------------------------------------------------------- test 1
def long_short_split(per):
    L = [t for v in per.values() for t in v if t[3] == 1]
    S = [t for v in per.values() for t in v if t[3] == -1]
    gl, nl = gross(L)
    gs, ns = gross(S)
    log("")
    log("  " + "=" * 66)
    log("  TEST 1  LONG vs SHORT -- is this a signal or is it drift?")
    log("  " + "=" * 66)
    log(f"    {'side':<8}{'trades':>9}{'gross/trade':>14}{'clears 0.170%':>16}")
    for lbl, g_, n_ in (("long", gl, nl), ("short", gs, ns)):
        log(f"    {lbl:<8}{n_:>9}{g_:>13.3f}%"
            f"{('YES' if g_ > BAR_MKT else 'no'):>16}")
    if not (np.isfinite(gl) and np.isfinite(gs)):
        return None
    log("")
    if gs > BAR_MKT and gl > BAR_MKT:
        log("    Both sides clear the cost bar. Shorts cannot collect upward")
        log("    drift, so a profitable short book is the strongest single")
        log("    piece of evidence that the ENTRY RULE is doing work.")
        return "both"
    if gl > BAR_MKT and gs <= 0:  # noqa: returns below carry the numbers
        log("    Longs earn it, shorts lose. That is what owning a rising")
        log("    market looks like. The entry rule is picking the direction")
        log("    the market was going anyway, and the hump is hold length.")
        return "long-only"
    log("    Mixed. Longs carry most of it but shorts are not bleeding, so")
    log("    the entry rule is contributing something beyond exposure --")
    log("    the null below decides how much.")
    return "mixed"


# ---------------------------------------------------------------- test 2
def halves(coins):
    log("")
    log("  " + "=" * 66)
    log("  TEST 2  DOES THE PEAK SIT IN THE SAME PLACE IN BOTH HALVES?")
    log("  " + "=" * 66)
    log("    The hump is the maximum of 11 correlated draws. Splitting the")
    log("    sample is the only cheap way to tell a market property from")
    log("    the shape of correlated noise.")
    log("")
    ref = max((V._load(f"{V.PERP}/{c}_1h.csv") for c in coins[:12]),
              key=lambda d: 0 if d is None else len(d))
    log(f"    {'tf':>5}{'1st half':>12}{'2nd half':>12}{'both >bar':>12}")
    rows = []
    for tf in HALVES_GRID:
        nb = len(ref) // tf
        mid = nb // 2
        p1, _, _ = collect(coins, tf, 0, mid)
        p2, _, _ = collect(coins, tf, mid, nb)
        g1, n1 = gross([t for v in p1.values() for t in v])
        g2, n2 = gross([t for v in p2.values() for t in v])
        both = "YES" if (g1 > BAR_MKT and g2 > BAR_MKT) else "no"
        log(f"    {tf:>4}h{g1:>11.3f}%{g2:>11.3f}%{both:>12}")
        rows.append((tf, g1, g2))
    a = np.array([r[1] for r in rows]); b = np.array([r[2] for r in rows])
    t = np.array([r[0] for r in rows])
    i1, i2 = int(np.nanargmax(a)), int(np.nanargmax(b))
    log("")
    log(f"    1st half peaks at {t[i1]}h, 2nd half peaks at {t[i2]}h")

    # THE TEST I WROTE FIRST WAS WRONG, and it passed on real data, which is
    # the worst way to find out. It compared only the LOCATION of the two
    # maxima -- 14h vs 16h, "SAME BAND", pass. But a maximum exists in every
    # series, including one that reads
    #     -0.149 -0.115 -0.190 -0.279 -0.018 +0.007 +0.372 -0.022
    # whose "peak" is the least-bad number in a set of zeros. Agreeing on
    # where the top of nothing is, is not replication. Argmax location is
    # scale-free and therefore cannot tell signal from noise.
    #
    # The test that means something: does the OTHER half still clear the
    # cost bar at the timeframe the FIRST half nominated? That is the
    # out-of-sample question an actual deployment would be asking.
    cross = [(t[i1], b[i1], "1st half's pick, scored in the 2nd"),
             (t[i2], a[i2], "2nd half's pick, scored in the 1st")]
    log("")
    for tf_, g_, why in cross:
        log(f"    {tf_:>3}h -> {g_:+.3f}%   ({why})"
            f"   {'clears' if g_ > BAR_MKT else 'DOES NOT CLEAR'}")
    agree = all(g_ > BAR_MKT for _, g_, _ in cross)
    band = [i for i, tf in enumerate(t) if 10 <= tf <= 16]
    ok = sum(1 for i in band if a[i] > BAR_MKT and b[i] > BAR_MKT)
    log(f"    timeframes in 10-16h clearing the bar in BOTH halves: "
        f"{ok}/{len(band)}")
    log("")
    if agree:
        log("    HOLDS UP. Each half's chosen timeframe still pays in the")
        log("    other. That is the result that would justify rebuilding.")
    else:
        log("    DOES NOT HOLD UP. The timeframe each half nominates does not")
        log("    pay in the other half. The full-sample hump is one half of")
        log("    the data carrying a flat half, which is indistinguishable")
        log("    from the edge having existed once and stopped.")
    return agree


# ---------------------------------------------------------------- test 3
def null(per, frames, btcs, sims, same_side, by_side=False):
    """Random-entry-bar control. Returns (real, draws, p, z) pooled, and if
    by_side also the same four for the long book and the short book alone.

    WHY BY SIDE. The pooled null is the wrong resolution once the long/short
    split is known. At 14h longs earn +0.025% (nothing) and shorts earn
    +1.058% (six times the cost bar), so the pooled number is an average of
    one dead book and one live one, and a pooled null tests neither. The
    question is specifically: does a RANDOMLY TIMED short earn what V29's
    short earns? The median coin in this universe fell 50.8%, so a short
    held 5-6 bars collects downward drift whenever it is opened. If the
    random short matches the real short, the Supertrend entry is decoration
    on "be short alts".
    """
    rng = np.random.default_rng(
        zlib.crc32(b"tfconfirm-same" if same_side else b"tfconfirm-rand"))
    allt = [t for v in per.values() for t in v]
    real_g, _ = gross(allt)
    real_l, _ = gross([t for t in allt if t[3] == 1])
    real_s, _ = gross([t for t in allt if t[3] == -1])
    draws, dl, ds = [], [], []
    for s in range(sims):
        acc = []
        for c, tr in per.items():
            df = frames[c]
            n = len(df)
            if n < 130:
                continue
            bars = rng.integers(60, n - 60, size=len(tr))
            if same_side:
                sides = [t[3] for t in tr]
            else:
                sides = rng.choice([1, -1], size=len(tr))
            acc += V.run_coin(
                c, df,
                entries_override=[(int(b), int(sd))
                                  for b, sd in zip(bars, sides)],
                btc=btcs.get(c))
        g_, _ = gross(acc)
        if np.isfinite(g_):
            draws.append(g_)
        if by_side:
            for bucket, side in ((dl, 1), (ds, -1)):
                gg, _ = gross([t for t in acc if t[3] == side])
                if np.isfinite(gg):
                    bucket.append(gg)
        if s == 0 or (s + 1) % max(1, sims // 8) == 0:
            log(f"      {s+1}/{sims}")

    def stat(real, arr):
        d = np.array(arr)
        if len(d) < 2:
            return real, d, float("nan"), float("nan")
        p = (1.0 + int((d >= real).sum())) / (1.0 + len(d))
        z = (real - d.mean()) / d.std() if d.std() > 0 else 0.0
        return real, d, p, z

    out = stat(real_g, draws)
    if not by_side:
        return out
    return out, stat(real_l, dl), stat(real_s, ds)


def benchmark_truth(coins):
    """The 'buy & hold' row V29 keeps losing to is a SUM across coins, and a
    sum is the wrong statistic for a universe with a 3000% outlier in it.
    Printed because every comparison in this project has been made against
    that number."""
    rs = []
    for c in coins:
        df = V._load(f"{V.PERP}/{c}_1h.csv")
        if df is None or len(df) < 100:
            continue
        rs.append(float(df["close"].values[-1] / df["open"].values[0] - 1))
    v = np.array(rs)
    order = np.argsort(-v)
    log("")
    log("  " + "=" * 66)
    log("  TEST 0  WHAT IS THE BENCHMARK, ACTUALLY?")
    log("  " + "=" * 66)
    log(f"    SUM of per-coin returns   {v.sum()*100:>9.1f}%  <- the"
        f" 'buy&hold' row")
    log(f"    MEAN per coin             {v.mean()*100:>9.1f}%")
    log(f"    MEDIAN per coin           {np.median(v)*100:>9.1f}%  <- the"
        f" typical coin")
    log(f"    coins down                {int((v < 0).sum())}/{len(v)}"
        f"   (down >50%: {int((v < -0.5).sum())})")
    log(f"    SUM excluding top 3       "
        f"{(v.sum()-v[order[:3]].sum())*100:>9.1f}%")
    log("")
    log("    The benchmark is one coin. Drop the three best and holding this")
    log("    universe LOSES money. So 'V29 does not beat buy-and-hold' was")
    log("    never the indictment it sounded like.")
    log(f"    The median coin fell {abs(np.median(v))*100:.0f}%, which SUGGESTS"
        f" a short held a few")
    log("    bars collects downward drift. That is a hypothesis, not a")
    log("    finding -- stops get hit by the upward spikes alts make on the")
    log("    way down, so it can go either way. The random-short line in")
    log("    test 3 measures it. Do not read a cause off this table.")
    return v


def short_deep(coins, tfs, sims):
    """The short book alone, split by time, against its own timing null.

    This exists because the pooled tests hid the answer. The whole-book
    halves sweep read ~0.00% in the second half at every timeframe and I
    nearly wrote V29 off on it. The second half is flat because the LONG
    book is noise large enough to cancel a short book that is not: at 12h
    shorts earn +0.857% in the first half and +1.331% in the second, both
    five times the cost bar, while longs earn nothing in either.

    Averaging a dead book with a live one and testing the average is a way
    to fail to find something that is there. Reported per side, per half.
    """
    ref = max((V._load(f"{V.PERP}/{c}_1h.csv") for c in coins[:12]),
              key=lambda d: 0 if d is None else len(d))
    log("")
    log("  " + "=" * 66)
    log("  TEST 4  THE SHORT BOOK ALONE, PER HALF, AGAINST ITS OWN NULL")
    log("  " + "=" * 66)
    log(f"    {'tf':>5}{'window':>10}{'n':>7}{'real':>10}{'null':>10}"
        f"{'sd':>8}{'z':>7}{'p':>8}")
    best = {}
    for tf in tfs:
        nb = len(ref) // tf
        mid = nb // 2
        cells = []
        for name, lo, hi in (("full", None, None), ("1st half", 0, mid),
                             ("2nd half", mid, nb)):
            per, frames, btcs = collect(coins, tf, lo, hi)
            real_t = [x for v in per.values() for x in v if x[3] == -1]
            real, _ = gross(real_t)
            rng = np.random.default_rng(zlib.crc32(f"sh{tf}{name}".encode()))
            d = []
            for _ in range(sims):
                acc = []
                for c, tr in per.items():
                    df = frames[c]
                    n = len(df)
                    k = sum(1 for x in tr if x[3] == -1)
                    if n < 130 or k == 0:
                        continue
                    acc += V.run_coin(
                        c, df, btc=btcs.get(c),
                        entries_override=[(int(b), -1) for b in
                                          rng.integers(60, n - 60, size=k)])
                g_, _ = gross(acc)
                if np.isfinite(g_):
                    d.append(g_)
            d = np.array(d)
            p = (1.0 + int((d >= real).sum())) / (1.0 + len(d))
            z = (real - d.mean()) / d.std() if d.std() > 0 else 0.0
            log(f"    {tf:>4}h{name:>10}{len(real_t):>7}{real:>9.3f}%"
                f"{d.mean():>9.3f}%{d.std():>7.3f}%{z:>7.2f}{p:>8.4f}")
            cells.append((name, real, p, z))
        best[tf] = cells
    # ---- multiplicity, stated rather than skipped -----------------------
    log("")
    log("    MULTIPLICITY. Finding this took 11 timeframes x 3 books = 33")
    log("    cells. The best nominal p must be read against that.")
    full_ps = [c[2] for tf in best for c in best[tf] if c[0] == "full"]
    if full_ps:
        bp = min(full_ps)
        b33, b7 = min(bp * 33, 1.0), min(bp * 7, 1.0)
        log(f"      best full-sample p            {bp:.4f}")
        log(f"      x33 (Bonferroni, harsh)       {b33:.3f}"
            f"   {'survives' if b33 <= 0.05 else 'does NOT survive'}")
        log(f"      x7  (timeframes are correlated resamples of the same")
        log(f"          bars, so ~3-4 independent speeds x 2 sides)"
            f"  {b7:.3f}   {'survives' if b7 <= 0.05 else 'does NOT survive'}")
        # These three lines used to read "Neither survives", hardcoded from
        # the snapshot I developed on, where the best p was 0.0365 and x7
        # gave 0.26. On a longer sample with a smaller p the sentence is
        # simply false, and a static verdict printed under live numbers is
        # the worst kind of wrong -- it looks computed. Now derived.
        if b33 <= 0.05:
            log("      Survives even the harsh correction. That is as close")
            log("      to a clean statistical result as this project has.")
        elif b7 <= 0.05:
            log("      Survives the realistic correction but not Bonferroni.")
            log("      Bonferroni assumes 33 independent tests and these are")
            log("      resamples of the same bars, so it is the wrong bound;")
            log("      it is quoted because picking the generous correction")
            log("      after seeing the answer is how people fool themselves.")
        else:
            log("      Does not survive either correction.")
        log("      Independent of any p-value: the sign agreeing in both")
        log("      halves at several adjacent speeds is replication, which")
        log("      is the evidence that matters here.")
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", type=int, default=14)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--skip-halves", action="store_true")
    ap.add_argument("--skip-null", action="store_true")
    ap.add_argument("--short-deep", action="store_true",
                    help="test 4: the short book per half vs its own null")
    a = ap.parse_args()

    if V.PERP is None:
        log("  cannot find taker_data; tried: " + ", ".join(V._TRIED))
        return 2
    coins = universe()
    if len(coins) < 10:
        log(f"  only {len(coins)} coins -- aborting")
        return 2

    # This line used to end "BTC MACD quarter-size ON" as a literal string.
    # The overlay is active only when BTC_1h.csv exists -- btc_macd_dir
    # returns None without it and every trade is full size. My snapshot has
    # no BTC file, so the label asserted a filter was running that was not,
    # on every run, for three commits. The third hardcoded claim in this
    # file after "the LONG book is noise" and "neither correction
    # survives": the defect is not any one string, it is writing
    # conclusions as literals in a report whose job is to carry
    # measurements.
    btc_on = os.path.exists(f"{V.PERP}/BTC_1h.csv")
    log(f"  V29 at {a.tf}h bars, {len(coins)} coins, long AND short")
    log(f"  BTC MACD quarter-size: "
        + ("ON" if btc_on else
           "OFF -- no BTC_1h.csv. This is the chop filter\n"
           "  added specifically to fix this strategy, so these results"
           " UNDERSTATE\n  V29. Fetch it: python3 fetch_taker.py --coin BTC"))
    log(f"  market-order round trip bar: {BAR_MKT:.3f}%")
    log("")
    per, frames, btcs = collect(coins, a.tf)
    g_all, n_all = gross([t for v in per.values() for t in v])
    pos = sum(1 for v in per.values()
              if np.isfinite(gross(v)[0]) and gross(v)[0] > 0)
    log(f"  full sample: {n_all} trades, gross {g_all:+.3f}%/trade, "
          f"{pos}/{len(per)} coins positive")

    verdicts = {}
    benchmark_truth(list(per))
    verdicts["split"] = long_short_split(per)
    _allt = [t for v in per.values() for t in v]
    verdicts["gl"] = gross([t for t in _allt if t[3] == 1])[0]
    verdicts["gs"] = gross([t for t in _allt if t[3] == -1])[0]

    if not a.skip_null:
        log("")
        log("  " + "=" * 66)
        log("  TEST 3  RANDOM-TIMING NULL -- does WHEN you enter matter?")
        log("  " + "=" * 66)
        log("    Same coins, same trade count, same side mix, same stop, same")
        log("    TP, same trailing exit. ONLY the entry bar is randomised.")
        log("    Reported for the whole book and for each side alone, because")
        log("    the two sides are not the same strategy.")
        log("")
        pooled, lng, sht = null(per, frames, btcs, a.sims, True, by_side=True)
        log("")
        log(f"      {'book':<8}{'real':>10}{'null mean':>12}{'null sd':>10}"
            f"{'z':>7}{'p':>9}")
        for lbl, (r_, d_, p_, z_) in (("all", pooled), ("long", lng),
                                      ("short", sht)):
            log(f"      {lbl:<8}{r_:>+9.3f}%{d_.mean():>+11.3f}%"
                f"{d_.std():>9.3f}%{z_:>7.2f}{p_:>9.4f}")
        verdicts["nullB"] = pooled[2]
        verdicts["nullB_short"] = sht[2]
        verdicts["null_long"] = float(lng[1].mean())
        verdicts["null_short"] = float(sht[1].mean())
        verdicts["zl"] = lng[3]
        log("")
        # This branch used to print "the universe fell and shorts held 5-6
        # bars collected that" whenever the short p was above 0.05. The
        # numbers printed two lines above REFUTE that sentence: the random
        # short earns a NEGATIVE number, so a randomly timed short in this
        # universe loses money and downward drift is not what the real
        # shorts are collecting. I keyed the explanation to a p-value and
        # asserted a cause the data in the same table contradicted. The
        # branch now describes the null's sign instead of narrating a story.
        nl_, ns_ = lng[1].mean(), sht[1].mean()
        log(f"      random LONG  earns {nl_:+.3f}%  -> "
            + ("upward drift IS available to a random entry"
               if nl_ > BAR_MKT else
               "a random long does NOT collect drift here"))
        log(f"      random SHORT earns {ns_:+.3f}%  -> "
            + ("downward drift IS available to a random entry"
               if ns_ > BAR_MKT else
               "a random short does NOT collect drift here"))
        if lng[0] <= lng[1].mean():
            log("      The real LONG book is WORSE than random entries. The")
            log("      long side is not a weak signal, it is noise, and it")
            log("      pays the full fee for the privilege.")
        log("")

    if not a.skip_halves:
        verdicts["halves"] = halves(coins)

    if a.short_deep:
        cells = short_deep(coins, (12, 14), max(a.sims, 300))
        ok = all(c[1] > BAR_MKT for tf in cells for c in cells[tf]
                 if c[0] != "full")
        verdicts["short_halves"] = ok

    # ---- the three-of-three test, stated before the run ------------------
    log("")
    log("  " + "=" * 66
        )
    log("  VERDICT")
    log("  " + "=" * 66)
    checks = []
    pB = verdicts.get("nullB")
    if pB is not None:
        checks.append((pB <= 0.05,
                       f"entry timing beats random, whole book (p={pB:.4f})"))
    pS = verdicts.get("nullB_short")
    if pS is not None:
        checks.append((pS <= 0.05,
                       f"entry timing beats random on the EARNING side "
                       f"(short, p={pS:.4f})"))
    hv = verdicts.get("halves")
    if hv is not None:
        checks.append((bool(hv), "whole-book timeframe replicates across "
                                 "halves"))
    sh = verdicts.get("short_halves")
    if sh is not None:
        checks.append((bool(sh), "SHORT book clears the cost bar in BOTH "
                                 "halves"))
    for ok, txt in checks:
        log(f"    [{'PASS' if ok else 'FAIL'}]  {txt}")
    # Everything below used to be static prose carrying the numbers from
    # the snapshot this file was written on -- including "The LONG book is
    # noise. It earns less than random entries", which was true there and
    # false on a longer sample where longs beat their null by +0.26pp.
    # A hardcoded conclusion printed beneath live numbers reads as though
    # it were computed from them. Derived now, or not claimed.
    log("")
    log("  WHAT THIS RUN SETTLES:")
    log("    - Fast trading is structurally dead: at 1h the gross edge is a")
    log("      fraction of the 0.170% round trip, over tens of thousands of")
    log("      trades. The overtrading was the problem.")
    log("    - The benchmark was never real: 'buy & hold' is a SUM over the")
    log("      universe dominated by its best name, and excluding the top 3")
    log("      holding it loses money. That comparison proved nothing.")
    gl_, gs_ = verdicts.get("gl"), verdicts.get("gs")
    nl_, ns_ = verdicts.get("null_long"), verdicts.get("null_short")
    if None not in (gl_, nl_):
        if gl_ <= nl_:
            log(f"    - The LONG book is noise: {gl_:+.3f}% against"
                f" {nl_:+.3f}% for random")
            log("      entries at the same times on the same coins. It is")
            log("      worse than random and pays the full fee.")
        elif gl_ > BAR_MKT:
            log(f"    - The LONG book clears the cost bar ({gl_:+.3f}%) but is"
                f" NOT")
            log(f"      distinguishable from random timing ({nl_:+.3f}%,"
                f" z={verdicts.get('zl', float('nan')):+.2f}).")
            log("      Treat it as exposure, not signal: it is the natural")
            log("      hedge against the short book's squeeze risk, which is")
            log("      a reason to keep it, but not evidence it works.")
        else:
            log(f"    - The LONG book does not clear the cost bar"
                f" ({gl_:+.3f}%).")
    if None not in (gs_, ns_):
        log(f"    - The SHORT book earns {gs_:+.3f}% while a randomly timed")
        log(f"      short earns {ns_:+.3f}%."
            + ("  A random short LOSES money here,"
               if ns_ < 0 else "  "))
        if ns_ < 0:
            log("      so the short result is NOT downward drift. The entry")
            log("      stack is avoiding the upward spikes that stop a short")
            log("      out -- it is a squeeze filter, which is exactly the")
            log("      'chop' problem the ATR and BTC-MACD overlays were")
            log("      added to fix.")
    log("")
    log("  THE RISK, which is not a statistical point:")
    log("    A short-only book across 60 alts is one position, not sixty.")
    log("    In an alt melt-up correlation goes to 1 and every leg loses")
    log("    together, in hours. Against a 5% daily and 10% total limit")
    log("    that is the single most likely way to fail an evaluation.")
    log("    This is a candidate sleeve with a hard gross-exposure cap, not")
    log("    a replacement book.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
