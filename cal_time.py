#!/usr/bin/env python3
"""
cal_time.py -- calendar levels re-scored honestly, and clock seasonality.

PART 1: CALENDAR LEVELS
ORCHESTRATOR_AND_CALENDAR.md reported prev-month-high break 1.66, prev-day-high
1.46, yearly open 1.28, prev-week-high 1.21 -- built on sleeves_book's hourly
mapping, which closes a stopped trade at the START of the bar that hits the
stop (a lookahead that took the FVG sleeve from 2.99 to 0.26). Never checked on
2023 either. Re-scored here with honest exits.

  levels    prev day high, prev week high, prev month high, yearly open
            (calendar_levels.levels(): completed periods only)
  signal    6h close crosses above the level (calendar_levels.signals break_up)
  entry     open of the next 6h bar
  stop      3 ATR(14) below entry, ATR from the signal bar; filled at the
            stop price, or at the bar's open if it gaps through
  exit      the stop, or the close of the 15th bar
  control   random long entries, same coins, same counts, same mechanics
  verdict   excess over random must be positive with monthly t > 2 on the
            2023 holdout; correlation with S/R break reported

PART 2: CLOCK SEASONALITY
  basket    equal-weight mean of all coins' hourly returns (63 coins)
  cells     7 weekdays (daily returns) and 24 UTC hours (hourly returns)
  pass      |t| above the Bonferroni line (2.69 for 7 cells, 3.04 for 24),
            same sign in both halves, and a mean bigger than the 0.17% round
            trip it would cost to trade it
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

import calendar_levels as CL

ROUND_TRIP = 0.0017
STOP_ATR = 3.0
HOLD = 15
FIT_START = pd.Timestamp("2024-02-13")
SIMS = 40
LEVELS = ["prev_day_high", "prev_wk_high", "prev_mo_high", "year_open"]


def atr(h, l, c, n=14):
    pc = np.concatenate([[np.nan], c[:-1]])
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    out = np.full(len(c), np.nan)
    if len(c) > n:
        out[n] = np.nanmean(tr[1:n + 1])
        for i in range(n + 1, len(c)):
            out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def trade(o, h, l, c, a, sig):
    """Long from o[sig+1]; ATR from the signal bar; honest stop fills."""
    eb = sig + 1
    n = len(c)
    if eb + 1 >= n or not np.isfinite(a[sig]) or a[sig] <= 0:
        return None
    ent = o[eb]
    stop = ent - STOP_ATR * a[sig]
    if stop <= 0:
        return None
    last = min(eb + HOLD - 1, n - 1)
    for j in range(eb, last + 1):
        if l[j] <= stop:
            fill = min(stop, o[j]) if j > eb else stop
            return fill / ent - 1 - ROUND_TRIP, (ent - stop) / ent, j
    return c[last] / ent - 1 - ROUND_TRIP, (ent - stop) / ent, last


def part1(coins, loader):
    rows = {lv: [] for lv in LEVELS}
    pools = {}
    for coin in coins:
        d = loader(coin, 6)
        if d is None or len(d) < 400:
            continue
        o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        a = atr(h, l, c)
        lv = CL.levels(d)
        for name in LEVELS:
            for s in CL.signals(d, lv, name, "break_up"):
                res = trade(o, h, l, c, a, s)
                if res:
                    rows[name].append((d.index[s + 1], d.index[res[2]], coin, res[0], res[1]))
        pools[coin] = (o, h, l, c, a, d.index)
    return {k: pd.DataFrame(v, columns=["ts", "exit", "coin", "pnl", "r"]).sort_values("ts")
            for k, v in rows.items()}, pools


def control(df, pools, rg):
    """Random entries: same coin, same count, same entry year-half, same mechanics."""
    out = []
    by = df.groupby(["coin", df.ts < FIT_START]).size().to_dict()
    for _ in range(SIMS):
        acc = {True: [], False: []}
        for (coin, pre), k in by.items():
            o, h, l, c, a, idx = pools[coin]
            ok = np.flatnonzero((idx < FIT_START) if pre else (idx >= FIT_START))
            ok = ok[(ok > 20) & (ok < len(c) - HOLD - 2)]
            if not len(ok):
                continue
            for s in ok[rg.integers(0, len(ok), k)]:
                res = trade(o, h, l, c, a, int(s))
                if res:
                    acc[pre].append(res[0] / res[1])
        out.append((np.mean(acc[True]) if acc[True] else np.nan,
                    np.mean(acc[False]) if acc[False] else np.nan))
    return np.array(out)


def mt(df):
    m = df.assign(R=df.pnl / df.r).groupby(df.ts.dt.to_period("M")).R.mean()
    if len(m) < 3:
        return np.nan, 0, len(m)
    return m.mean() / (m.std(ddof=1) / np.sqrt(len(m))), int((m > 0).sum()), len(m)


def part2(coins, loader):
    rets = {}
    for coin in coins:
        d = loader(coin, 1)
        if d is not None and len(d) > 1000:
            rets[coin] = d["close"].pct_change()
    R = pd.DataFrame(rets)
    R = R[R.index >= R.index[0] + pd.Timedelta(days=30)]
    basket = R.mean(axis=1, skipna=True).dropna()
    basket = basket[basket.abs() < 0.5]           # drop data glitches, not moves
    mid = basket.index[len(basket) // 2]

    def cells(series, key, labels, bonf):
        print(f"  {'cell':<8}{'n':>6}{'mean':>10}{'t':>7}{'half 1':>10}{'half 2':>10}  verdict")
        found = []
        for k, lab in labels:
            x = series[key(series.index) == k]
            if len(x) < 30:
                continue
            t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
            a, b = x[x.index < mid].mean(), x[x.index >= mid].mean()
            sig = abs(t) > bonf and np.sign(a) == np.sign(b) == np.sign(x.mean())
            tradable = abs(x.mean()) > ROUND_TRIP
            v = "PASS" if sig and tradable else ("real but below costs" if sig else "")
            if v == "PASS":
                found.append(lab)
            print(f"  {lab:<8}{len(x):>6}{x.mean()*100:>9.3f}%{t:>7.2f}{a*100:>9.3f}%{b*100:>9.3f}%  {v}")
        return found

    daily = (1 + basket).resample("D").prod() - 1
    daily = daily[daily != 0]
    print("\n  Day of week (UTC), daily basket return  -- Bonferroni |t| > 2.69")
    d_found = cells(daily, lambda i: i.dayofweek,
                    list(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])), 2.69)
    print("\n  Hour of day (UTC), hourly basket return -- Bonferroni |t| > 3.04")
    h_found = cells(basket, lambda i: i.hour, [(h, f"{h:02d}:00") for h in range(24)], 3.04)
    return d_found, h_found


def main() -> int:
    import event_audit as EA
    import short_risk as SR
    coins = SR.universe()
    rg = np.random.default_rng(zlib.crc32(b"cal_time"))

    print("\n  PART 1 -- calendar levels, honest exits, 6h, long on break")
    print(f"  {'level':<15}{'n':>6}{'win':>5}{'R/trade':>9}{'rand PRE':>9}{'real PRE':>9}"
          f"{'exc PRE':>8}{'t PRE':>7}{'rand FIT':>9}{'real FIT':>9}{'exc FIT':>8}{'t FIT':>7}")
    trades, pools = part1(coins, EA.load_bars)
    try:
        sr = pd.read_csv("ps_oos_fixed_daily.csv", index_col=0, parse_dates=True)["sr_6h"]
    except Exception:
        sr = None
    for name, df in trades.items():
        if len(df) < 50:
            print(f"  {name:<15}{len(df):>6}  too few")
            continue
        R = df.pnl / df.r
        ctrl = control(df, pools, rg)
        pre, fit = df[df.ts < FIT_START], df[df.ts >= FIT_START]
        rp, rf = (pre.pnl / pre.r).mean(), (fit.pnl / fit.r).mean()
        cp, cf = np.nanmean(ctrl[:, 0]), np.nanmean(ctrl[:, 1])
        tp, tf = mt(pre)[0], mt(fit)[0]
        print(f"  {name:<15}{len(df):>6}{(R > 0).mean()*100:>4.0f}%{R.mean():>+9.3f}"
              f"{cp:>+9.3f}{rp:>+9.3f}{rp - cp:>+8.3f}{tp:>7.2f}"
              f"{cf:>+9.3f}{rf:>+9.3f}{rf - cf:>+8.3f}{tf:>7.2f}")
        if sr is not None:
            dd = df.groupby(df.exit.dt.floor("D")).pnl.sum()
            j = pd.concat([dd, sr], axis=1).fillna(0.0)
            j = j[(j.index >= sr.index[60]) & (j.index <= sr.index[-1])]
            print(f"  {'':<15}  correlation with S/R break (daily): {j.corr().iloc[0, 1]:+.2f}")

    print("\n  PART 2 -- clock seasonality, equal-weight basket of", len(coins), "coins")
    d_found, h_found = part2(coins, EA.load_bars)
    print(f"\n  tradeable clock effects: days {d_found or 'none'}, hours {h_found or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
