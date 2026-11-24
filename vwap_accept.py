#!/usr/bin/env python3
"""
vwap_accept.py -- does "acceptance outside value" pay on crypto perps?

THE IDEA (auction-market / MMT style). Session VWAP marks fair value; its +/-1
standard-deviation bands mark the value area. When price closes outside value
several bars in a row, the market has ACCEPTED the new price, and should
travel on to the next level rather than snap back. That makes it a
continuation trade -- the one shape that has held up repeatedly in this
project (FVG direction, OI + aggressive buying, closing flows continuing).

PRE-REGISTERED -- written before the first run, not changed afterwards
  session      UTC day; VWAP and bands reset at 00:00 UTC
  value area   session VWAP +/- 1 volume-weighted standard deviation
  warm-up      no signals in the first 2 hours of the session
  acceptance   3 consecutive closes above the upper band -> long,
               3 consecutive closes below the lower band -> short
               first signal per direction per session only
  entry        open of the next bar (never the close that confirmed it)
  stop         a close back through VWAP (acceptance failed); the level used
               is the PREVIOUS bar's VWAP, so the stop never uses the bar's
               own data before it is printed
  exit         the stop, or the last bar of the session
  costs        0.17% round trip; 1R must exceed 3x that (as all day)
  universes    15m bars, 25 coins, Feb 2025 - Aug 2026
               1h bars, 63 coins, Feb 2023 - Sep 2026
  verdict      excess over random entries (same coin, side, session hours,
               stop and exit rule), monthly-clustered t > 2, and both halves
               of the sample positive. Anything less is a rejection.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

import of_flow as OF

ROUND_TRIP = 0.0017
MIN_R = 3 * ROUND_TRIP
ACCEPT_BARS = 3
WARMUP_HOURS = 2
SIMS = 40


def sessions(d: pd.DataFrame):
    """Per-bar session VWAP and band width, cumulative within each UTC day.
    Values at bar i use bars up to and including i (known at i's close)."""
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    v = d["volume"].clip(lower=0)
    day = d.index.floor("D")
    cv = v.groupby(day).cumsum()
    vwap = (tp * v).groupby(day).cumsum() / cv.replace(0, np.nan)
    m2 = (tp * tp * v).groupby(day).cumsum() / cv.replace(0, np.nan)
    sd = np.sqrt((m2 - vwap * vwap).clip(lower=0))
    hour = (d.index - day).total_seconds() / 3600.0
    return vwap.values, sd.values, day.values, np.asarray(hour)


def trade(o, h, l, c, vwap, day, eb, side):
    """Enter at o[eb]; stop on a close back through the previous bar's VWAP;
    otherwise exit at the session's last bar. Returns (pnl, r) or None."""
    n = len(c)
    if eb >= n or day[eb] != day[eb - 1]:
        return None
    ent = o[eb]
    ref = vwap[eb - 1]
    if not (np.isfinite(ent) and np.isfinite(ref)) or ent <= 0:
        return None
    r = (ent - ref) / ent * side
    if not (r > MIN_R):
        return None
    j = eb
    while j < n and day[j] == day[eb]:
        stop = vwap[j - 1]
        if np.isfinite(stop) and ((side > 0 and c[j] < stop) or (side < 0 and c[j] > stop)):
            return side * (c[j] / ent - 1) - ROUND_TRIP, r
        j += 1
    return side * (c[j - 1] / ent - 1) - ROUND_TRIP, r


def signals(c, vwap, sd, day, hour, bar_h):
    out = []
    n = len(c)
    up = dn = 0
    done = {1: None, -1: None}
    for i in range(1, n - 1):
        if day[i] != day[i - 1]:
            up = dn = 0
            done = {1: None, -1: None}
        if hour[i] + bar_h <= WARMUP_HOURS or not np.isfinite(sd[i]) or sd[i] <= 0:
            continue
        up = up + 1 if c[i] > vwap[i] + sd[i] else 0
        dn = dn + 1 if c[i] < vwap[i] - sd[i] else 0
        for side, cnt in ((1, up), (-1, dn)):
            if cnt == ACCEPT_BARS and done[side] != day[i] and day[i + 1] == day[i]:
                out.append((i + 1, side))
                done[side] = day[i]
    return out


def run(coins, loader, tf, bar_h):
    rows, null_pool = [], {}
    for coin in coins:
        d = loader(coin, tf)
        if d is None or len(d) < 500:
            continue
        o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
        vwap, sd, day, hour = sessions(d)
        for eb, side in signals(c, vwap, sd, day, hour, bar_h):
            res = trade(o, h, l, c, vwap, day, eb, side)
            if res:
                rows.append((d.index[eb], coin, side, res[0], res[1]))
        # random-entry pool: any bar past warm-up with session left, either side
        eligible = np.flatnonzero((hour + bar_h > WARMUP_HOURS) & (hour < 23 - bar_h))
        null_pool[coin] = (o, h, l, c, vwap, day, eligible)
    return pd.DataFrame(rows, columns=["ts", "coin", "side", "pnl", "r"]).sort_values("ts"), null_pool


def null(trades, pool, rg):
    means = []
    by = trades.groupby(["coin", "side"]).size().to_dict()
    for _ in range(SIMS):
        acc = []
        for (coin, side), k in by.items():
            o, h, l, c, vwap, day, elig = pool[coin]
            got, tries = 0, 0
            while got < k and tries < k * 40:
                tries += 1
                eb = int(elig[rg.integers(0, len(elig))])
                res = trade(o, h, l, c, vwap, day, eb, side)
                if res:
                    acc.append(res[0] / res[1])
                    got += 1
        means.append(np.mean(acc) if acc else 0.0)
    return np.array(means)


def report(label, d, pool, rg):
    if len(d) < 50:
        print(f"  {label}: only {len(d)} trades -- no verdict possible")
        return
    R = d.pnl / d.r
    mon = d.assign(R=R).groupby(d.ts.dt.to_period("M")).R.mean()
    tm = mon.mean() / (mon.std(ddof=1) / np.sqrt(len(mon)))
    nl = null(d, pool, rg)
    half = d.ts.iloc[len(d) // 2]
    h1, h2 = R[d.ts < half].mean(), R[d.ts >= half].mean()
    exc = R.mean() - nl.mean()
    ok = tm > 2 and exc > 0 and h1 > 0 and h2 > 0
    print(f"  {label}")
    print(f"    trades {len(d):,}  ({(d.side > 0).sum():,} long / {(d.side < 0).sum():,} short)"
          f"   median 1R {d.r.median()*100:.2f}%   win {(R > 0).mean()*100:.0f}%")
    print(f"    mean R {R.mean():+.3f}   random entries {nl.mean():+.3f} (sd {nl.std():.3f})"
          f"   excess {exc:+.3f}   per trade {d.pnl.mean()*100:+.3f}%")
    print(f"    monthly t {tm:+.2f}  ({int((mon > 0).sum())}/{len(mon)} months up)"
          f"   halves {h1:+.3f} / {h2:+.3f}")
    for s, lab in ((1, "long "), (-1, "short")):
        x = R[d.side == s]
        if len(x) > 30:
            print(f"      {lab} n={len(x):>6,}  mean R {x.mean():+.3f}")
    print(f"    verdict: {'PASS' if ok else 'REJECT'}\n")


def main() -> int:
    import event_audit as EA
    import short_risk as SR
    rg = np.random.default_rng(zlib.crc32(b"vwap_accept"))
    print("\n  VWAP acceptance -- pre-registered, one run per universe\n")
    d, pool = run(OF.coins5(), OF.load_flow5, 15, 0.25)
    report("15m bars, 25 coins, Feb 2025 - Aug 2026", d, pool, rg)
    d, pool = run(SR.universe(), EA.load_bars, 1, 1.0)
    report("1h bars, 63 coins, Feb 2023 - Sep 2026", d, pool, rg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
