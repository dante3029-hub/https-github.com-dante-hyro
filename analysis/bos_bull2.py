#!/usr/bin/env python3
"""
Fair-test the BOS long side before rejecting it.

1. BTC macro-regime composition of the window and of each half.
2. Long-side performance conditional on BTC regime (bull days only).
3. TP/stop sweep for the long side -- the short side's 3-ATR TP was tuned on
   shorts; longs may simply need different exits.
4. Regime-gated 'both' variant: shorts in bear, longs in bull.
"""
import sys, csv, os, datetime as dt
import numpy as np
sys.path.insert(0, "/home/user/workspace")
sys.path.insert(0, "/tmp")

from reference_impl import (select_universe, build_matrices, run_sleeve, sig_delta,
                            sig_relvol, nz, W_CORE, W_BOS)
from bos_bull import run_bos_dir, sharpe, ANN
import bos_bull

WS = "/home/user/workspace"


def btc_daily_close():
    """Daily closes for BTC from the 1h panel (BTC is excluded from the traded
    universe but is the macro regime reference)."""
    for cand in (f"{WS}/clean_panel/hist/BTC_1h.csv", f"{WS}/run/hist/BTC_1h.csv"):
        if os.path.exists(cand):
            path = cand
            break
    else:
        return None
    day = {}
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        next(r, None)
        for row in r:
            if len(row) < 6 or not row[1].strip():
                continue
            try:
                t = int(row[1]); c = float(row[5])
            except ValueError:
                continue
            day[dt.datetime.fromtimestamp(t / 1000, tz=dt.timezone.utc).date()] = c
    return day


def regime_mask(cal, sma=200):
    """True = BTC bull (close > SMA200)."""
    d = btc_daily_close()
    if not d:
        return None
    days = sorted(d)
    px = np.array([d[x] for x in days], float)
    ma = np.full(len(px), np.nan)
    for i in range(sma - 1, len(px)):
        ma[i] = px[i - sma + 1:i + 1].mean()
    lut = {days[i]: (px[i] > ma[i]) if not np.isnan(ma[i]) else None
           for i in range(len(days))}
    return np.array([bool(lut.get(c, False)) for c in cal])


def main():
    coins = select_universe()
    dates, PX, R, FN, DN, RV = build_matrices(coins)
    cal = []
    d = dates[45]
    while d <= dates[-1]:
        cal.append(d); d += dt.timedelta(days=1)
    cal = np.array(cal); T = len(cal); h = T // 2

    bull = regime_mask(cal)
    print("=" * 70)
    print("1. BTC MACRO REGIME COMPOSITION (close vs SMA200)")
    print("=" * 70)
    print(f"  full window : {bull.mean():6.1%} bull days  ({bull.sum()}/{T})")
    print(f"  1st half    : {bull[:h].mean():6.1%} bull   ({bull[:h].sum()}/{h})")
    print(f"  2nd half    : {bull[h:].mean():6.1%} bull   ({bull[h:].sum()}/{T-h})")

    s_short = np.load("/tmp/bos_short.npy")
    s_long = np.load("/tmp/bos_long.npy")

    print("\n" + "=" * 70)
    print("2. PERFORMANCE CONDITIONAL ON BTC REGIME")
    print("=" * 70)
    for name, s in (("short", s_short), ("long", s_long)):
        sb, sr = s[bull], s[~bull]
        print(f"  BOS {name:5s}  bull days: Sharpe {sharpe(sb):7.4f}  sum {sb.sum():8.4f}  n={len(sb)}")
        print(f"  BOS {name:5s}  bear days: Sharpe {sharpe(sr):7.4f}  sum {sr.sum():8.4f}  n={len(sr)}")

    print("\n" + "=" * 70)
    print("3. LONG-SIDE TP/STOP SWEEP (is 3-ATR TP just wrong for longs?)")
    print("=" * 70)
    print(f"  {'stop':>5s} {'tp':>5s} {'Sharpe':>9s} {'1st':>9s} {'2nd':>9s} {'sum':>9s}")
    best = None
    for stop_a in (1.5, 2.0, 3.0):
        for tp_a in (2.0, 3.0, 5.0, 8.0):
            bos_bull.BOS_ATR_STOP = stop_a
            bos_bull.BOS_TP_ATR = tp_a
            import reference_impl as ri
            ri.BOS_ATR_STOP = stop_a; ri.BOS_TP_ATR = tp_a
            s, _, _, _ = run_bos_dir(coins, cal, "long")
            f, a, b = sharpe(s), sharpe(s[:h]), sharpe(s[h:])
            print(f"  {stop_a:5.1f} {tp_a:5.1f} {f:9.4f} {a:9.4f} {b:9.4f} {s.sum():9.4f}")
            if best is None or f > best[0]:
                best = (f, stop_a, tp_a, a, b)
    print(f"\n  best long config: stop={best[1]} tp={best[2]}  "
          f"FULL={best[0]:.4f}  1st={best[3]:.4f}  2nd={best[4]:.4f}")
    print("  NOTE: this is an IN-SAMPLE grid search over 12 configs. Treat the")
    print("  best number as an upper bound inflated by selection, not an estimate.")

    # restore
    import reference_impl as ri
    ri.BOS_ATR_STOP = 2.0; ri.BOS_TP_ATR = 3.0
    bos_bull.BOS_ATR_STOP = 2.0; bos_bull.BOS_TP_ATR = 3.0

    print("\n" + "=" * 70)
    print("4. REGIME-GATED: longs only on BTC-bull days, shorts only on bear")
    print("=" * 70)
    gated = np.where(bull, s_long, s_short)
    print(f"  gated       Sharpe FULL={sharpe(gated):.4f}  1st={sharpe(gated[:h]):.4f}  2nd={sharpe(gated[h:]):.4f}")
    short_only = s_short
    print(f"  short-only  Sharpe FULL={sharpe(short_only):.4f}  1st={sharpe(short_only[:h]):.4f}  2nd={sharpe(short_only[h:]):.4f}")
    add = s_short + s_long
    print(f"  short+long  Sharpe FULL={sharpe(add):.4f}  1st={sharpe(add[:h]):.4f}  2nd={sharpe(add[h:]):.4f}")

    print("\n" + "=" * 70)
    print("5. WALK-FORWARD ON THE FULL BLEND (decide on 1st half, test on 2nd)")
    print("=" * 70)
    delta = run_sleeve(sig_delta(DN), dates, PX, R, FN)
    relvol = run_sleeve(sig_relvol(RV), dates, PX, R, FN)
    core = (nz(delta) + nz(relvol)) / 2
    L = min(len(core), T); c2 = core[-L:]; hh_ = L // 2
    s_both = np.load("/tmp/bos_both.npy")
    for name, b in (("short-only (current)", s_short), ("both", s_both),
                    ("regime-gated", gated)):
        b2 = b[-L:]
        blend = W_CORE * nz(c2) + W_BOS * nz(b2)
        print(f"  {name:22s} blend 1st={sharpe(blend[:hh_]):7.4f}   2nd={sharpe(blend[hh_:]):7.4f}")


if __name__ == "__main__":
    main()
