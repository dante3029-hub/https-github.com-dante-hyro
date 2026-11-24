#!/usr/bin/env python3
"""
mtf_bos.py -- Dante's Monday-level setup, multi-timeframe version.
Pre-registered; one run per cell. Written before the first run.

  "use a bigger time frame for market structure but use lower time frames"

WHY: the 1h-only version (level_bos.py) put the stop at the last 1h swing,
median 4.8% away, so 2R needed ~9.6% in 48h. Only 16% got there. Direction
from the higher timeframe + entry/stop from the lower one should tighten the
stop so the target is reachable.

RULES (short side; longs mirror on the Monday high)
  1 LEVEL     this week's Monday low (UTC), usable Tue-Sun
  2 REVISIT   a lower-timeframe bar trades at/below the Monday low
  3 HTF BOS   within 48h of the revisit, a 4h candle CLOSES below the latest
              confirmed 4h swing low (3-bar fractal, confirmed 3 bars later)
              -> short bias, valid 48h from that 4h close
  4 LTF ENTRY after the 4h break: price must build NEW lower-timeframe
              structure (a swing low confirmed after the 4h break), then a
              lower-timeframe candle closes below it -> short next bar's open
  5 STOP      latest confirmed lower-timeframe swing high at the entry signal
  6 EXIT      2R target, stop, or 48 hours; stop+target same bar = stop
  7 COSTS     0.17% round trip; 1R must exceed 0.51% or the trade is skipped
  one trade per side per week

CELLS
  A  4h structure, 1h entry   top-30 coins by volume, Feb 2023 - Sep 2026
  B  4h structure, 15m entry  25 coins with 15m data, Feb 2025 - Aug 2026

CONTROL  random bars, same coin/side, same LTF structural stop, same exits
VERDICT  excess over random > 0, monthly t > 2, both halves positive
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

import level_bos as LB

K = 3
HTF = "4h"
WINDOW_H = 48
HOLD_H = 48


def swings_conf(h, l, k=K):
    """Latest confirmed swing high/low at each bar AND the bar it was
    confirmed on. Pivot at i is confirmed at i+k (no lookahead)."""
    n = len(h)
    sh = np.full(n, np.nan); sl = np.full(n, np.nan)
    shc = np.full(n, -1); slc = np.full(n, -1)
    ph, pl = {}, {}
    for i in range(k, n - k):
        if h[i] == h[i - k:i + k + 1].max():
            ph.setdefault(i + k, h[i])
        if l[i] == l[i - k:i + k + 1].min():
            pl.setdefault(i + k, l[i])
    lh = ll = np.nan; lhc = llc = -1
    for t in range(n):
        if t in ph: lh, lhc = ph[t], t
        if t in pl: ll, llc = pl[t], t
        sh[t], sl[t], shc[t], slc[t] = lh, ll, lhc, llc
    return sh, sl, shc, slc


def htf_events(d, bar):
    """4h breaks of structure, mapped to the LTF bar whose CLOSE is the 4h close."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    cnt = d["close"].resample(HTF).count()
    full = int(pd.Timedelta(HTF) / bar)
    h4 = d.resample(HTF).agg(agg)[cnt == full].dropna()
    sh4, sl4 = LB.swings(h4.high.values, h4.low.values)
    c4 = h4.close.values
    close_t = h4.index + pd.Timedelta(HTF)           # when the 4h bar is known
    ltf_close = d.index + bar
    pos = ltf_close.get_indexer(close_t)
    bear = np.zeros(len(d), bool); bull = np.zeros(len(d), bool)
    for j in range(1, len(h4)):
        p = pos[j]
        if p < 0:
            continue
        if np.isfinite(sl4[j - 1]) and c4[j] < sl4[j - 1]:
            bear[p] = True
        if np.isfinite(sh4[j - 1]) and c4[j] > sh4[j - 1]:
            bull[p] = True
    return bear, bull


def setups(d, bar):
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    hi, lo, wk = LB.levels(d, "monday")
    sh, sl, shc, slc = swings_conf(h, l)
    bear4, bull4 = htf_events(d, bar)
    per_h = int(pd.Timedelta("1h") / bar)
    win, n = WINDOW_H * per_h, len(c)
    out, used = [], set()
    rev = {}        # side -> bar the level was revisited
    bias = {}       # side -> bar of the 4h break
    for t in range(1, n - 1):
        if not (np.isfinite(hi[t]) and np.isfinite(lo[t])):
            rev.clear(); bias.clear()
            continue
        if l[t] <= lo[t]:
            rev[-1] = t
        if h[t] >= hi[t]:
            rev[1] = t
        # 4h break after (or on the bar of) the revisit, within the window
        for side, ev in ((-1, bear4), (1, bull4)):
            if ev[t] and side in rev and t - rev[side] <= win:
                bias[side] = t
        for side, b in list(bias.items()):
            if t - b > win:
                del bias[side]
                continue
            if t <= b or (wk[t], side) in used:
                continue
            if side < 0:
                hit = slc[t - 1] > b and c[t] < sl[t - 1]
                stop = sh[t]
            else:
                hit = shc[t - 1] > b and c[t] > sh[t - 1]
                stop = sl[t]
            if hit and np.isfinite(stop):
                out.append((t + 1, side, stop))
                used.add((wk[t], side))
                del bias[side]
                rev.pop(side, None)
    return out


def run_cell(label, coins, loader, tf_arg, bar, rg):
    per_h = int(pd.Timedelta("1h") / bar)
    LB.HOLD = HOLD_H * per_h                     # trade() and control() read this
    rows, pool = [], {}
    for coin in coins:
        d = loader(coin, tf_arg)
        if d is None or len(d) < 200 * per_h:
            continue
        o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
        for eb, side, stop in setups(d, bar):
            res = LB.trade(o, h, l, c, eb, side, stop)
            if res:
                rows.append((d.index[eb], coin, side, res[0], res[1]))
        sh, sl, _, _ = swings_conf(h, l)
        pool[coin] = (o, h, l, c, sh, sl)
    df = pd.DataFrame(rows, columns=["ts", "coin", "side", "pnl", "r"]).sort_values("ts")
    if len(df) < 100:
        print(f"  {label}: only {len(df)} trades, no verdict")
        return
    R = df.pnl / df.r
    gross = (df.pnl + LB.ROUND_TRIP) / df.r
    mon = df.assign(R=R).groupby(df.ts.dt.to_period("M")).R.mean()
    tm = mon.mean() / (mon.std(ddof=1) / np.sqrt(len(mon)))
    cm = np.nanmean(LB.control(df, pool, rg))
    mid = df.ts.iloc[len(df) // 2]
    h1, h2 = R[df.ts < mid].mean(), R[df.ts >= mid].mean()
    tgt = (np.abs(gross - LB.TARGET_R) < 1e-9).mean()
    stp = (gross <= -0.999).mean()
    exc = R.mean() - cm
    ok = exc > 0 and tm > 2 and h1 > 0 and h2 > 0
    print(f"  {label}")
    print(f"    trades {len(df):,} ({(df.side>0).sum():,} long / {(df.side<0).sum():,} short)"
          f"   median stop {df.r.median()*100:.2f}%   win {(R>0).mean()*100:.0f}%")
    print(f"    hit 2R {tgt*100:.0f}%   stopped {stp*100:.0f}%   timed out {(1-tgt-stp)*100:.0f}%")
    print(f"    R/trade gross {gross.mean():+.3f}  net {R.mean():+.3f}   random {cm:+.3f}   excess {exc:+.3f}")
    print(f"    monthly t {tm:+.2f} ({int((mon>0).sum())}/{len(mon)} months up)   halves {h1:+.3f} / {h2:+.3f}"
          f"   long {R[df.side>0].mean():+.3f}  short {R[df.side<0].mean():+.3f}")
    print(f"    verdict: {'PASS' if ok else 'REJECT'}\n")


def main() -> int:
    import event_audit as EA
    import short_risk as SR
    import of_flow as OF
    import fvg_revisit as FV
    rg = np.random.default_rng(zlib.crc32(b"mtf_bos"))
    print("\n  Monday level + 4h structure + lower-timeframe entry -- pre-registered\n")
    top30 = FV.top_by_liquidity(SR.universe(), EA.load_bars, 30)
    run_cell("A  4h structure, 1h entry, top 30, Feb 2023 - Sep 2026",
             top30, EA.load_bars, 1, pd.Timedelta("1h"), rg)
    run_cell("B  4h structure, 15m entry, 25 coins, Feb 2025 - Aug 2026",
             OF.coins5(), OF.load_flow5, 15, pd.Timedelta("15min"), rg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
