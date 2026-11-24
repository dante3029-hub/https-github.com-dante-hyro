#!/usr/bin/env python3
"""
pa_ltf.py -- the book's own timeframe pairing: H1 levels, M15 entries.

Page 53: "The pairing is completely up to you and your style, so it
could be weekly into daily, H4 into H1 or H1 into M15... If you work a
9 to 5 then this all still applies on higher timeframes. H1 is probably
the best fit."
Page 53 again: "use higher timeframes to make decisions on lower
timeframes, but never use a lower timeframe to make a decision about a
higher one."

So H1 supplies everything that is a DECISION -- the range, the
direction, the target zone, and the 2R gate. M15 supplies only the
ENTRY. That division is the rule, not a preference.

WHY THE ENTRY MOVES THE NEEDLE (p.30): the same setup entered at the
level instead of at the breakout close is "an over 3R difference in the
outcome", because the stop is the far side of the range you entered
from, and the M15 range is a fraction of the H1 range. Same target,
smaller 1R, more R.

WHAT WOULD MAKE THIS REAL: more R per trade than the H1-only version
AND a sample big enough to mean something AND stability across halves.
The H1-only run already produced +0.113R on 286 continuations that
decayed to zero in the second half, and a 4h reversal cell of n=38 whose
sign flipped at 1h. Both are warnings, not results.

DATA. 5m bars for 25 coins, Feb 2025 - Aug 2026. H1 and M15 are both
resampled from the same 5m file so the two series cannot drift out of
alignment -- an H1 bar is exactly four M15 bars, every time.

USAGE
    python3 pa_ltf.py
    python3 pa_ltf.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import zlib

import numpy as np
import pandas as pd

import pa_book as P
import pa_target as PT

log = logging.getLogger("pa_ltf")

M5DIR = ("/tmp/claude-0/-home-claude/45f2f086-9843-5124-bddd-90b82b67de76"
         "/scratchpad/m5raw")
_CACHE: dict = {}


def coins():
    return sorted(os.path.basename(p).replace("_5m.csv", "")
                  for p in glob.glob(f"{M5DIR}/*_5m.csv"))


def load5(coin):
    """5m bars, cached. Validated on the way in: monotonic timestamps,
    no duplicate bars, no non-positive prices, high >= low."""
    if coin in _CACHE:
        return _CACHE[coin]
    p = f"{M5DIR}/{coin}_5m.csv"
    try:
        d = pd.read_csv(p, usecols=["timestamp_ms", "open", "high",
                                    "low", "close", "volume"])
    except Exception as e:
        log.warning("  %s unreadable: %s", coin, e)
        return None
    d["ts"] = pd.to_datetime(d["timestamp_ms"], unit="ms")
    d = d.set_index("ts").sort_index()
    d = d[~d.index.duplicated(keep="first")]
    for c in ("open", "high", "low", "close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]
    d = d[(d.high >= d.low) & (d.high >= d.open) & (d.high >= d.close)
          & (d.low <= d.open) & (d.low <= d.close)]
    _CACHE[coin] = d if len(d) > 5000 else None
    return _CACHE[coin]


def resample(d, minutes):
    """Resample 5m -> `minutes`, dropping INCOMPLETE buckets.

    Resampling floors to the bucket boundary, so a file starting at
    08:15 produces an "08:00" hourly bar built from three 5m bars rather
    than twelve. That bar's open is wrong and its range is understated,
    and it sits at the very start of the series where the first ranges
    get drawn. Only buckets with the full complement of 5m bars are
    kept.
    """
    if d is None:
        return None
    g = d.resample(f"{minutes}min")
    r = g.agg({"open": "first", "high": "max", "low": "min",
               "close": "last", "volume": "sum"}).dropna()
    full = g.size()
    r = r[full.reindex(r.index).fillna(0) == minutes // 5]
    return r if len(r) > 400 else None


def run(reversal, use_ltf=True, ltf_min=15, htf_min=60):
    """H1 decides, M15 enters.

    use_ltf=False runs the identical trade list with the H1 breakout
    entry instead, so the ONLY difference between the two columns is
    where the entry came from. Anything else and the comparison is not a
    comparison.
    """
    need = PT.MIN_R_REV if reversal else PT.MIN_R_CONT
    step = htf_min // ltf_min
    rows = []
    for coin in coins():
        d5 = load5(coin)
        dh = resample(d5, htf_min)
        dl = resample(d5, ltf_min)
        if dh is None or dl is None:
            continue
        o, h, l, c = (dh["open"].values, dh["high"].values,
                      dh["low"].values, dh["close"].values)
        lo_, lh, ll, lc = (dl["open"].values, dl["high"].values,
                           dl["low"].values, dl["close"].values)
        ranges = PT.sparse(P.find_ranges(o, h, l, c))
        for rng in ranges:
            s = P.signals(o, h, l, c, rng, reversal)
            if s is None or s[0] >= len(c):
                continue
            eb, side = s
            # ---- the DECISION, all of it on H1
            ent_h = o[eb]
            stop_h = rng.lo if side > 0 else rng.hi
            r_h = abs(ent_h - stop_h) / ent_h
            if r_h <= 0 or not np.isfinite(r_h):
                continue
            tz = PT.next_zone(ranges, eb, side, ent_h, rng)
            if tz is None:
                continue
            if abs(tz - ent_h) / ent_h / r_h < need:
                continue                       # "I don't take it"

            if not use_ltf:
                res = PT.simulate_to_zone(o, h, l, c, eb, side, rng, tz)
                if res is None:
                    continue
                rows.append((dh.index[eb], coin, side,
                             res[0], res[1], res[2]))
                continue

            # ---- the ENTRY, and only the entry, on M15
            t0 = dh.index[eb]
            j = int(dl.index.searchsorted(t0))
            if j < 60 or j >= len(lc) - 10:
                continue
            seg = slice(max(0, j - 400), min(len(lc), j + P.MAX_WAIT * step))
            so, sh_, sl_, sc = lo_[seg], lh[seg], ll[seg], lc[seg]
            off = max(0, j - 400)
            lr = PT.sparse(P.find_ranges(so, sh_, sl_, sc))
            got = None
            for r2 in lr:
                if r2.start + off < j:
                    continue                   # must form AFTER the H1 call
                s2 = P.signals(so, sh_, sl_, sc, r2, reversal)
                if s2 is None:
                    continue
                eb2, side2 = s2
                if side2 != side:
                    continue                   # HTF decides direction
                if eb2 + off < j or eb2 >= len(sc):
                    continue
                got = (eb2, r2)
                break
            if got is None:
                continue
            eb2, r2 = got
            ent_l = so[eb2]
            stop_l = r2.lo if side > 0 else r2.hi
            r_l = abs(ent_l - stop_l) / ent_l
            if r_l <= 0 or not np.isfinite(r_l):
                continue
            # the H1 target still has to be worth >= the book's minimum
            # FROM THE NEW ENTRY, or the better price bought nothing
            if abs(tz - ent_l) / ent_l / r_l < need:
                continue
            if side * (tz - ent_l) <= 0:
                continue                       # target already passed
            res = PT.simulate_to_zone(so, sh_, sl_, sc, eb2, side, r2, tz)
            if res is None:
                continue
            rows.append((dl.index[off + eb2], coin, side,
                         res[0], res[1], res[2]))
    return pd.DataFrame(rows, columns=["ts", "coin", "side", "pnl",
                                       "bars", "r"]).sort_values("ts")


def null(d, sims, rg, ltf_min):
    """Random entry bars on the SAME bars, same side mix, same 1R."""
    if not len(d):
        return np.array([0.0])
    per = {}
    for coin in d.coin.unique():
        per[coin] = resample(load5(coin), ltf_min)
    counts = d.groupby("coin").size().to_dict()
    sides = d.side.values
    rmed = float(np.median(d.r.values))
    out = []
    for _ in range(sims):
        acc = []
        for coin, k in counts.items():
            dd = per.get(coin)
            if dd is None:
                continue
            o, h, l, c = (dd["open"].values, dd["high"].values,
                          dd["low"].values, dd["close"].values)
            n = len(c)
            for _i in range(k):
                eb = int(rg.integers(60, n - 60))
                sd = int(rg.choice(sides))
                w = rmed * c[eb]
                rr = P.Rng(lo=c[eb] - w, hi=c[eb] + w, start=eb,
                           trend_up=sd > 0)
                tgt = c[eb] * (1 + sd * PT.MIN_R_CONT * rmed)
                res = PT.simulate_to_zone(o, h, l, c, eb, sd, rr, tgt)
                if res is not None:
                    acc.append(res[0])
        if acc:
            out.append(float(np.mean(acc)))
    return np.array(out) if out else np.array([0.0])


def selftest() -> int:
    cs = coins()
    print(f"  {len(cs)} coins with 5m data")
    d5 = load5(cs[0])
    assert d5 is not None, "5m load failed"
    h1, m15 = resample(d5, 60), resample(d5, 15)
    print(f"  {cs[0]}: 5m={len(d5)} H1={len(h1)} M15={len(m15)}")
    assert abs(len(m15) / max(len(h1), 1) - 4) < 0.1, "M15/H1 is not 4:1"
    assert h1.index[0] >= d5.index[0] and h1.index[-1] <= d5.index[-1]
    # an H1 bar must equal the aggregate of its four M15 bars
    t = h1.index[500]
    blk = m15.loc[t:t + pd.Timedelta(minutes=45)]
    assert len(blk) == 4, "H1 bar does not cover exactly four M15 bars"
    assert abs(blk.high.max() - h1.high.loc[t]) < 1e-9, "high mismatch"
    assert abs(blk.low.min() - h1.low.loc[t]) < 1e-9, "low mismatch"
    print("  H1 and M15 agree bar for bar: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.selftest:
        return selftest()
    rg = np.random.default_rng(zlib.crc32(b"pa_ltf"))
    print("\n  H1 levels / M15 entries -- the book's own pairing (p.53)")
    print("  H1 decides range, direction, target and the 2R gate.")
    print("  M15 supplies the entry only.\n")
    print(f"  {'setup':<28}{'n':>6}{'1R':>7}{'win%':>6}{'R/trade':>9}"
          f"{'nullR':>8}{'edgeR':>8}{'p':>7}{'H1':>8}{'H2':>8}")
    for rev in (False, True):
        for ltf in (False, True):
            d = run(rev, use_ltf=ltf)
            nm = "reversal" if rev else "continuation"
            lbl = f"{nm}, {'M15 entry' if ltf else 'H1 entry'}"
            if len(d) < 30:
                print(f"  {lbl:<28}{len(d):>6}   too few trades")
                continue
            R = d.pnl / d.r
            cut = d.ts.iloc[len(d) // 2]
            h1 = float((d[d.ts < cut].pnl / d[d.ts < cut].r).mean())
            h2 = float((d[d.ts >= cut].pnl / d[d.ts >= cut].r).mean())
            rmed = float(d.r.median())
            nl = null(d, a.sims, rg, 15 if ltf else 60)
            nlR = nl / rmed
            p = float((nl >= d.pnl.mean()).mean())
            print(f"  {lbl:<28}{len(d):>6}{rmed*100:>6.2f}%"
                  f"{(R>0).mean()*100:>5.0f}%{R.mean():>9.3f}"
                  f"{nlR.mean():>8.3f}{R.mean()-nlR.mean():>8.3f}"
                  f"{p:>7.3f}{h1:>8.3f}{h2:>8.3f}")
    print("\n  M15 entry should show a SMALLER 1R and MORE R per trade")
    print("  than the H1 entry on the same decisions (p.30). If it does")
    print("  not, the lower timeframe is adding noise, not precision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
