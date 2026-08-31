#!/usr/bin/env python3
"""
Is a BULL (long) side worth adding to the BOS sleeve?

The existing code already computes `state == 1` (higher highs AND higher lows)
and then never trades it. This tests the exact mirror of the short rule.

SHORT (existing):  state == -1, close breaks BELOW last swing low,
                   taker delta < 0, stop = ent + 2*ATR, tp = ent - 3*ATR,
                   invalidate on close ABOVE last swing high.
LONG  (mirror):    state == +1, close breaks ABOVE last swing high,
                   taker delta > 0, stop = ent - 2*ATR, tp = ent + 3*ATR,
                   invalidate on close BELOW last swing low.

Everything else identical: 4h bars, k=5 pivots, 30-bar time exit, 6 concurrent
slots, 2 x FEE per round trip, zero-filled daily series.
"""
import sys, datetime as dt
import numpy as np
sys.path.insert(0, "/home/user/workspace")

from reference_impl import (select_universe, build_matrices, run_sleeve,
                            sig_delta, sig_relvol, nz, _atr, _pivots, _load_4h,
                            BOS_PIVOT_K, BOS_ATR_STOP, BOS_TP_ATR, BOS_EXIT_BAR,
                            BOS_MAXPOS, FEE, W_CORE, W_BOS, D)

ANN = np.sqrt(365.0)


def sharpe(x):
    x = np.asarray(x, float)
    s = x.std()
    return float(x.mean() / s * ANN) if s > 0 else 0.0


def run_bos_dir(coins, calendar, direction="short", maxpos=BOS_MAXPOS):
    """direction: 'short' | 'long' | 'both'. Concurrency cap is shared."""
    trades = []
    stats = {"short": 0, "long": 0}
    for coin in coins:
        d4 = _load_4h(coin)
        if d4 is None:
            continue
        ts, o, hh, ll, cl, vv, tb = d4
        n = len(cl)
        A = _atr(hh, ll, cl)
        ph, pl = _pivots(hh, ll, BOS_PIVOT_K)
        dn = np.divide(2 * tb - vv, np.where(vv > 0, vv, np.nan))
        lastH = lastL = prevH = prevL = np.nan
        state = 0
        i = BOS_PIVOT_K + 20
        while i < n - 1:
            if not np.isnan(ph[i]):
                prevH, lastH = lastH, ph[i]
            if not np.isnan(pl[i]):
                prevL, lastL = lastL, pl[i]
            if not any(np.isnan(v) for v in (lastH, prevH, lastL, prevL)):
                if lastH > prevH and lastL > prevL:
                    state = 1
                elif lastH < prevH and lastL < prevL:
                    state = -1

            want_s = direction in ("short", "both")
            want_l = direction in ("long", "both")

            brk_dn = (not np.isnan(lastL)) and cl[i] < lastL and cl[i - 1] >= lastL
            trig_s = (want_s and brk_dn and state == -1
                      and (not np.isnan(dn[i])) and dn[i] < 0)

            brk_up = (not np.isnan(lastH)) and cl[i] > lastH and cl[i - 1] <= lastH
            trig_l = (want_l and brk_up and state == 1
                      and (not np.isnan(dn[i])) and dn[i] > 0)

            if (trig_s or trig_l) and not np.isnan(A[i]) and A[i] > 0:
                ent = o[i + 1]
                j = i + 1
                px = None
                if trig_s:
                    stop = ent + BOS_ATR_STOP * A[i]
                    tp = ent - BOS_TP_ATR * A[i]
                    while j < min(i + 1 + BOS_EXIT_BAR, n):
                        if hh[j] >= stop: px = stop; break
                        if ll[j] <= tp:   px = tp;   break
                        if not np.isnan(lastH) and cl[j] > lastH:
                            px = o[min(j + 1, n - 1)]; break
                        j += 1
                    if px is None:
                        px = cl[min(j, n - 1)]
                    pnl = (ent - px) / ent - 2 * FEE
                    stats["short"] += 1
                else:
                    stop = ent - BOS_ATR_STOP * A[i]
                    tp = ent + BOS_TP_ATR * A[i]
                    while j < min(i + 1 + BOS_EXIT_BAR, n):
                        if ll[j] <= stop: px = stop; break
                        if hh[j] >= tp:   px = tp;   break
                        if not np.isnan(lastL) and cl[j] < lastL:
                            px = o[min(j + 1, n - 1)]; break
                        j += 1
                    if px is None:
                        px = cl[min(j, n - 1)]
                    pnl = (px - ent) / ent - 2 * FEE
                    stats["long"] += 1

                trades.append((ts[i + 1], ts[min(j, n - 1)], pnl))
                i = j + 1
                continue
            i += 1

    trades.sort()
    live, taken, daily = [], [], {}
    for entry_ts, exit_ts, pnl in trades:
        live = [x for x in live if x > entry_ts]
        if len(live) < maxpos:
            live.append(exit_ts)
            taken.append((exit_ts, pnl))
    for exit_ts, pnl in taken:
        d = D(exit_ts)
        daily[d] = daily.get(d, 0.0) + pnl
    series = np.array([daily.get(d, 0.0) for d in calendar])
    return series, stats, len(taken), len(trades)


def main():
    coins = select_universe()
    print(f"universe: {len(coins)} coins")
    dates, PX, R, FN, DN, RV = build_matrices(coins)

    cal = []
    d = dates[45]
    while d <= dates[-1]:
        cal.append(d)
        d += dt.timedelta(days=1)
    cal = np.array(cal)
    T = len(cal)
    h = T // 2
    print(f"calendar: {T} days, {cal[0]} .. {cal[-1]}  (half at {cal[h]})\n")

    results = {}
    for direction in ("short", "long", "both"):
        s, st, taken, gen = run_bos_dir(coins, cal, direction)
        results[direction] = s
        nz_days = int((s != 0).sum())
        print(f"--- BOS {direction.upper():5s} ---")
        print(f"  trades generated={gen:5d}  taken after 6-slot cap={taken:5d}  "
              f"(short={st['short']}, long={st['long']})")
        print(f"  active days={nz_days}/{T} ({nz_days/T:.1%})")
        print(f"  Sharpe FULL = {sharpe(s):.4f}   "
              f"1st half = {sharpe(s[:h]):.4f}   2nd half = {sharpe(s[h:]):.4f}")
        print(f"  total return (sum of daily) = {s.sum():.4f}   "
              f"daily std = {s.std():.6f}   worst day = {s.min():.4f}\n")

    np.save("/tmp/bos_short.npy", results["short"])
    np.save("/tmp/bos_long.npy", results["long"])
    np.save("/tmp/bos_both.npy", results["both"])

    # correlation between the two legs
    a, b = results["short"], results["long"]
    if a.std() > 0 and b.std() > 0:
        print(f"corr(short, long) = {np.corrcoef(a, b)[0,1]:.4f}\n")

    # ---- effect on the actual blend --------------------------------------
    delta = run_sleeve(sig_delta(DN), dates, PX, R, FN)
    relvol = run_sleeve(sig_relvol(RV), dates, PX, R, FN)
    core = (nz(delta) + nz(relvol)) / 2
    L = min(len(core), T)
    c2 = core[-L:]
    print("--- effect on the Strategy B blend (2/3 core + 1/3 bos) ---")
    for direction in ("short", "long", "both"):
        b2 = results[direction][-L:]
        blend = W_CORE * nz(c2) + W_BOS * nz(b2)
        hh_ = L // 2
        print(f"  bos={direction:5s}  blend Sharpe FULL={sharpe(blend):.4f}  "
              f"1st={sharpe(blend[:hh_]):.4f}  2nd={sharpe(blend[hh_:]):.4f}")
    np.save("/tmp/bos_core.npy", c2)


if __name__ == "__main__":
    main()
