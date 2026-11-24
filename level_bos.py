#!/usr/bin/env python3
"""
level_bos.py -- Dante's setup: revisit a key level, then market structure
breaks, then trade. Pre-registered; one run.

  "if we revisit Monday low and market structure is broken take a short,
   vice versa"

LEVELS (known in advance, never from the bar they are used on)
  monday     this week's Monday high and low (UTC), used Tue-Sun
  prevweek   last week's high and low, used all week

STRUCTURE (1h bars)
  swing high/low: 3-bar fractal, CONFIRMED only 3 bars after it prints
  bearish BOS:    a close below the latest swing low confirmed before this bar
  bullish BOS:    a close above the latest swing high confirmed before this bar

CELLS (all fixed before the first run)
  continuation  revisit LOW  -> bearish BOS within 24 bars -> SHORT   (Dante's)
                revisit HIGH -> bullish BOS within 24 bars -> LONG    (vice versa)
  reversal      revisit LOW  -> bullish BOS within 24 bars -> LONG    (sweep & reclaim)
                revisit HIGH -> bearish BOS within 24 bars -> SHORT
  x 2 level sets = 4 cells. One trade per side per level per week.

TRADE
  entry  open of the bar after the BOS close
  stop   the opposite swing point confirmed at the BOS bar (structural)
  exit   2R target or 48 bars; bar touching stop and target = stop
  costs  0.17% round trip; 1R must exceed 3x that

CONTROL   random bars on the same coin, same side, same structural stop,
          same exit: isolates the TIMING the setup chooses
VERDICT   excess over random > 0, monthly t > 2, both halves positive
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

ROUND_TRIP = 0.0017
MIN_R = 3 * ROUND_TRIP
K = 3
WINDOW = 24
HOLD = 48
TARGET_R = 2.0
SIMS = 30


def swings(h, l, k=K):
    """Latest confirmed swing high/low AS KNOWN at each bar (no lookahead)."""
    n = len(h)
    sh_at = np.full(n, np.nan)
    sl_at = np.full(n, np.nan)
    last_h = last_l = np.nan
    pend_h, pend_l = {}, {}
    for i in range(k, n - k):
        if h[i] == h[i - k:i + k + 1].max():
            pend_h.setdefault(i + k, h[i])
        if l[i] == l[i - k:i + k + 1].min():
            pend_l.setdefault(i + k, l[i])
    for t in range(n):
        if t in pend_h:
            last_h = pend_h[t]
        if t in pend_l:
            last_l = pend_l[t]
        sh_at[t] = last_h
        sl_at[t] = last_l
    return sh_at, sl_at


def levels(d, which):
    idx = d.index
    wk = idx.to_period("W")
    if which == "monday":
        mon = d[idx.dayofweek == 0]
        mh = mon["high"].groupby(mon.index.to_period("W")).max()
        ml = mon["low"].groupby(mon.index.to_period("W")).min()
        hi = pd.Series(wk.map(mh), index=idx).astype(float)
        lo = pd.Series(wk.map(ml), index=idx).astype(float)
        hi[idx.dayofweek == 0] = np.nan            # not known until Monday closes
        lo[idx.dayofweek == 0] = np.nan
    else:
        wh = d["high"].groupby(wk).max().shift(1)
        wl = d["low"].groupby(wk).min().shift(1)
        hi = pd.Series(wk.map(wh), index=idx).astype(float)
        lo = pd.Series(wk.map(wl), index=idx).astype(float)
    return hi.values, lo.values, np.asarray(wk.astype(str))


def trade(o, h, l, c, eb, side, stop):
    n = len(c)
    if eb >= n - 1:
        return None
    ent = o[eb]
    r = (stop - ent) / ent * (-side)
    if not np.isfinite(r) or r <= MIN_R:
        return None
    tgt = ent * (1 + side * TARGET_R * r)
    for j in range(eb, min(eb + HOLD, n)):
        if (side > 0 and l[j] <= stop) or (side < 0 and h[j] >= stop):
            fill = stop if j == eb else (min(stop, o[j]) if side > 0 else max(stop, o[j]))
            return side * (fill / ent - 1) - ROUND_TRIP, r
        if (side > 0 and h[j] >= tgt) or (side < 0 and l[j] <= tgt):
            return side * (tgt / ent - 1) - ROUND_TRIP, r
    last = min(eb + HOLD, n) - 1
    return side * (c[last] / ent - 1) - ROUND_TRIP, r


def setups(d, which, mode):
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    hi, lo, wk = levels(d, which)
    sh, sl = swings(h, l)
    n = len(c)
    out = []
    used = set()
    armed = {}                       # (level side) -> bar armed
    for t in range(1, n - 1):
        if not (np.isfinite(hi[t]) and np.isfinite(lo[t])):
            continue
        if l[t] <= lo[t]:
            armed["low"] = t
        if h[t] >= hi[t]:
            armed["high"] = t
        bear = np.isfinite(sl[t - 1]) and c[t] < sl[t - 1]
        bull = np.isfinite(sh[t - 1]) and c[t] > sh[t - 1]
        for lvl, a in list(armed.items()):
            if t - a > WINDOW:
                del armed[lvl]
                continue
            if mode == "continuation":
                side = -1 if (lvl == "low" and bear) else 1 if (lvl == "high" and bull) else 0
            else:
                side = 1 if (lvl == "low" and bull) else -1 if (lvl == "high" and bear) else 0
            if not side:
                continue
            key = (wk[t], lvl, side)
            if key in used:
                continue
            stop = sh[t] if side < 0 else sl[t]
            out.append((t + 1, side, stop))
            used.add(key)
            del armed[lvl]
    return out


def run(coins, loader, which, mode):
    rows, pool = [], {}
    for coin in coins:
        d = loader(coin, 1)
        if d is None or len(d) < 2000:
            continue
        o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
        sh, sl = swings(h, l)
        for eb, side, stop in setups(d, which, mode):
            res = trade(o, h, l, c, eb, side, stop)
            if res:
                rows.append((d.index[eb], coin, side, res[0], res[1]))
        pool[coin] = (o, h, l, c, sh, sl)
    return pd.DataFrame(rows, columns=["ts", "coin", "side", "pnl", "r"]).sort_values("ts"), pool


def control(df, pool, rg):
    by = df.groupby(["coin", "side"]).size().to_dict()
    out = []
    for _ in range(SIMS):
        acc = []
        for (coin, side), k in by.items():
            o, h, l, c, sh, sl = pool[coin]
            n, got, tries = len(c), 0, 0
            while got < k and tries < 40 * k:
                tries += 1
                t = int(rg.integers(200, n - HOLD - 2))
                stop = sh[t] if side < 0 else sl[t]
                if not np.isfinite(stop) or (side < 0 and stop <= o[t + 1]) or (side > 0 and stop >= o[t + 1]):
                    continue
                res = trade(o, h, l, c, t + 1, side, stop)
                if res:
                    acc.append(res[0] / res[1])
                    got += 1
        out.append(np.mean(acc) if acc else np.nan)
    return np.array(out)


def main() -> int:
    import event_audit as EA
    import short_risk as SR
    coins = SR.universe()
    rg = np.random.default_rng(zlib.crc32(b"level_bos"))
    print("\n  Level revisit + break of structure -- pre-registered, 4 cells, 1h, 63 coins\n")
    print(f"  {'cell':<28}{'n':>6}{'win':>5}{'R':>8}{'random':>8}{'excess':>8}"
          f"{'t_mon':>7}{'mon+':>7}{'half1':>8}{'half2':>8}{'long R':>8}{'short R':>8}  verdict")
    for which in ("monday", "prevweek"):
        for mode in ("continuation", "reversal"):
            df, pool = run(coins, EA.load_bars, which, mode)
            lab = f"{which} / {mode}"
            if len(df) < 100:
                print(f"  {lab:<28}{len(df):>6}  too few")
                continue
            R = df.pnl / df.r
            mon = df.assign(R=R).groupby(df.ts.dt.to_period("M")).R.mean()
            tm = mon.mean() / (mon.std(ddof=1) / np.sqrt(len(mon)))
            ctrl = control(df, pool, rg)
            cm = np.nanmean(ctrl)
            mid = df.ts.iloc[len(df) // 2]
            h1, h2 = R[df.ts < mid].mean(), R[df.ts >= mid].mean()
            exc = R.mean() - cm
            ok = exc > 0 and tm > 2 and h1 > 0 and h2 > 0
            print(f"  {lab:<28}{len(df):>6}{(R > 0).mean()*100:>4.0f}%{R.mean():>+8.3f}{cm:>+8.3f}"
                  f"{exc:>+8.3f}{tm:>7.2f}{int((mon > 0).sum()):>4}/{len(mon):<2}{h1:>+8.3f}{h2:>+8.3f}"
                  f"{R[df.side > 0].mean():>+8.3f}{R[df.side < 0].mean():>+8.3f}  {'PASS' if ok else 'REJECT'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
