#!/usr/bin/env python3
"""
ps_oos.py -- the book's price-structure sleeves (fvg 12h, sr 6h, pattern 6h):
how much is lookahead, and does what is left survive 2023?

THE BUG. Every event sleeve in sleeves_book.py ends a stopped trade like this:

    for j in range(eb + 1, ...):
        if lo[j] <= stp:          # bar j's LOW hits the stop
            ex = j; break
    tr.append((s, d.index[eb], d.index[ex], side))

Resampled bars are labelled by their START. So the position is closed at the
start of bar j -- before the move inside bar j that hit the stop. The trade is
exited at the open of the bar that ruins it, a price the strategy could only
have chosen by knowing the bar's low in advance. On a 12h bar that skips up to
eleven hours of the worst move of the trade.

THE FIX. A trade stopped in bar j is held THROUGH bar j (exit at its end). On
the hourly grid that includes the whole adverse bar, which can be slightly
worse than a stop fill at the exact stop price -- conservative, deliberately.
Time exits (no stop) are unchanged.

THE TEST. As-built vs fixed, full sample. Then the fixed version against a
random-timing control (same coins, same counts, same sides, same durations,
random entry hours), excess over the control by period:
    PRE  2023 -> 2024-02-12   never used by the book
    FIT  2024-02-13 -> 2026-08-22
"""
from __future__ import annotations

import os
import sys
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sleeves_book as sb   # noqa: E402

sb.TAKER = os.path.join(HERE, "taker_data")
sb.OIDIR = os.path.join(HERE, "oi_data")
FIT_START = pd.Timestamp("2024-02-13")
FIT_END = pd.Timestamp("2026-08-22")
SEEDS = 12

CONFIGS = {   # BOOK_VERIFIED.md configs: (function, tf hours, hold bars, kwargs)
    "fvg_12h": (sb.fvg_hourly, 12, 10, dict(tf=12, stop_atr=1.5, hold=10)),
    "sr_6h": (sb.sr_hourly, 6, 15, dict(tf=6, stop_atr=3.0, hold=15)),
    "pattern_6h": (sb.pattern_hourly, 6, 15, dict(tf=6, stop_atr=2.0, hold=15)),
}


def capture_trades(fn, coins, hidx, cols, kw):
    """Run a sleeve function and intercept the trade list it builds."""
    box = {}
    real = sb.positions_to_hourly

    def spy(trades, *a, **k):
        box["tr"] = list(trades)
        return real(trades, *a, **k)

    sb.positions_to_hourly = spy
    try:
        fn(coins, hidx, cols, **kw)
    finally:
        sb.positions_to_hourly = real
    return box.get("tr", [])


def fix_exits(trades, tf, hold):
    """Stopped trades (shorter than the time exit) are held through the bar
    that hit the stop instead of being closed at its start."""
    out, n_fixed = [], 0
    full = pd.Timedelta(hours=tf * hold)
    for coin, t0, t1, side in trades:
        if (t1 - t0) < full:
            t1 = t1 + pd.Timedelta(hours=tf)
            n_fixed += 1
        out.append((coin, t0, t1, side))
    return out, n_fixed


def to_daily(trades, hidx, cols, RH):
    W = sb.positions_to_hourly(trades, hidx, cols, per_position=1.0 / 6,
                               max_gross=1.0)
    g = (W.shift(1) * RH).sum(axis=1)
    c = W.diff().abs().sum(axis=1) * sb.FEE
    return (g - c).fillna(0.0).resample("D").sum()


def randomize(trades, hidx, first_valid, seed):
    """Same coin, side and duration; entry hour drawn uniformly from the
    coin's own history. Destroys timing, keeps everything else."""
    rg = np.random.default_rng(seed)
    out = []
    last = hidx[-1]
    for coin, t0, t1, side in trades:
        dur = t1 - t0
        lo = first_valid.get(coin, hidx[0])
        hi = last - dur
        if hi <= lo:
            continue
        span = int((hi - lo) / pd.Timedelta(hours=1))
        s = lo + pd.Timedelta(hours=int(rg.integers(0, max(span, 1))))
        out.append((coin, s, s + dur, side))
    return out


def sharpe(x):
    x = x.dropna()
    return float(x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else 0.0


def monthly_t(x):
    s = x.resample("ME").sum()
    s = s[s != 0]
    if len(s) < 3 or s.std(ddof=1) == 0:
        return 0.0, len(s), 0
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))), len(s), int((s > 0).sum())


def main() -> int:
    coins = [c for c in sb.CORE24
             if os.path.exists(os.path.join(sb.TAKER, f"{c}_1h.csv"))]
    hidx, PX, RH = sb.hourly_panel(coins)
    cols = list(PX.columns)
    first_valid = {c: PX[c].first_valid_index() for c in cols}
    warm = hidx[0] + pd.Timedelta(days=60)

    print("\n  1. How much of each sleeve is the stop-bar lookahead?\n")
    print(f"  {'sleeve':<12}{'trades':>7}{'stopped':>9}{'as-built':>10}{'fixed':>8}")
    fixed_tr = {}
    for name, (fn, tf, hold, kw) in CONFIGS.items():
        tr = capture_trades(fn, coins, hidx, cols, kw)
        ftr, nf = fix_exits(tr, tf, hold)
        fixed_tr[name] = ftr
        a = to_daily(tr, hidx, cols, RH)
        b = to_daily(ftr, hidx, cols, RH)
        a, b = a[a.index >= warm], b[b.index >= warm]
        print(f"  {name:<12}{len(tr):>7}{nf:>9}{sharpe(a):>10.2f}{sharpe(b):>8.2f}")

    print("\n  2. Fixed sleeves vs random-timing control, excess by period\n")
    print(f"  {'sleeve':<12}{'period':<6}{'real Sh':>8}{'ctrl Sh':>8}"
          f"{'excess Sh':>10}{'t_mon':>7}{'mon+':>7}")
    store = {}
    for name, ftr in fixed_tr.items():
        real = to_daily(ftr, hidx, cols, RH)
        ctrl = pd.concat([to_daily(randomize(ftr, hidx, first_valid,
                                             zlib.crc32(f"{name}{s}".encode())),
                                   hidx, cols, RH) for s in range(SEEDS)], axis=1)
        cm = ctrl.mean(axis=1).reindex(real.index).fillna(0.0)
        exc = real - cm
        store[name] = real
        for pn, m in (("PRE", (real.index >= warm) & (real.index < FIT_START)),
                      ("FIT", (real.index >= FIT_START) & (real.index <= FIT_END))):
            tm, nm, mp = monthly_t(exc[m])
            print(f"  {name:<12}{pn:<6}{sharpe(real[m]):>8.2f}{sharpe(cm[m]):>8.2f}"
                  f"{sharpe(exc[m]):>10.2f}{tm:>7.2f}{mp:>4}/{nm}")
        print()
    pd.DataFrame(store).to_csv(os.path.join(HERE, "ps_oos_fixed_daily.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
