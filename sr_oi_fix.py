#!/usr/bin/env python3
"""sr_oi_fix.py -- S/R break 6h: remove the OI-filter lookahead.

THE BUG. sleeves_book.oi_up resamples hourly OI into 6h bins with .last(); pandas
labels a bin by its START, so oi_up[bar eb] is the OI change over bar eb, known
only at eb's CLOSE. The trade enters at eb's OPEN. The filter therefore asks
"did OI rise during the next six hours?" -- and OI rises with price.

THE FIX. Shift by one bin: at eb's open the bot knows the change over the
signal bar i (ended at eb's open). Everything else unchanged (stop-bar fix from
ps_oos applied). Same random-timing control and periods as ps_oos.py.
"""
import os, zlib
import numpy as np, pandas as pd
import sleeves_book as sb
import ps_oos as P

orig = sb.oi_up
def oi_up_fixed(sym, tf):
    s = orig(sym, tf)
    return None if s is None else s.shift(1).fillna(False).astype(bool)

def run(label):
    coins = [c for c in sb.CORE24 if os.path.exists(os.path.join(sb.TAKER, f"{c}_1h.csv"))]
    hidx, PX, RH = sb.hourly_panel(coins)
    cols = list(PX.columns); first_valid = {c: PX[c].first_valid_index() for c in cols}
    warm = hidx[0] + pd.Timedelta(days=60)
    fn, tf, hold, kw = P.CONFIGS["sr_6h"]
    tr = P.capture_trades(fn, coins, hidx, cols, kw)
    ftr, _ = P.fix_exits(tr, tf, hold)
    real = P.to_daily(ftr, hidx, cols, RH)
    ctrl = pd.concat([P.to_daily(P.randomize(ftr, hidx, first_valid, zlib.crc32(f"sr_6h{s}".encode())),
                                 hidx, cols, RH) for s in range(P.SEEDS)], axis=1)
    cm = ctrl.mean(axis=1).reindex(real.index).fillna(0.0); exc = real - cm
    print(f"  {label}: {len(ftr)} trades")
    for pn, m in (("PRE (2023, unseen)", (real.index >= warm) & (real.index < P.FIT_START)),
                  ("FIT", (real.index >= P.FIT_START) & (real.index <= P.FIT_END)),
                  ("ALL", real.index >= warm)):
        tm, nm, mp = P.monthly_t(exc[m])
        print(f"    {pn:<20} real Sharpe {P.sharpe(real[m]):5.2f}  random {P.sharpe(cm[m]):5.2f}"
              f"  excess {P.sharpe(exc[m]):5.2f}  t_mon {tm:5.2f}  {mp}/{nm} months")
    return real

print("\n  S/R break 6h -- OI filter as built vs without lookahead\n")
a = run("AS TESTED (OI checked at end of entry bar)")
sb.oi_up = oi_up_fixed
b = run("FIXED (OI checked at the signal bar, known at entry)")
pd.DataFrame({"sr_6h_asbuilt": a, "sr_6h_oifix": b}).to_csv("sr_oi_fix_daily.csv")
