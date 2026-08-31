#!/usr/bin/env python3
"""
Issue #5: is MAX_SLEEVE_MULTIPLIER = 1.0 a safe placeholder or a live handbrake?

The cap clips (1/3)/vol_product BEFORE the wB scale-down and extra_factor.
It was never in the backtest. Two questions:

  Q1 Does it BIND on real history? If it never binds, live sizing matches the
     backtest and 1.0 is free insurance.
  Q2 If it does bind, how much notional is it removing, and does removing the
     cap change realized risk?

Method: walk the real sleeve histories forward, recompute multipliers at each
date with cap=1.0 vs cap=1e9 (effectively uncapped), compare.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/user/workspace")

from bot.sleeve_history import compute_sleeve_histories
from portfolio_layer.position_sizer import compute_sleeve_multipliers
from portfolio_layer.ab_blend import DEFAULT_WINDOW

W = DEFAULT_WINDOW
print(f"window = {W}")

h = compute_sleeve_histories()
keys = ("main", "short", "flow", "delta", "relvol", "bos")
n = min(len(h[k]) for k in keys)
print("history lengths:", {k: len(h[k]) for k in keys}, "-> using", n)
H = {k: np.asarray(h[k], float)[-n:] for k in keys}

SLEEVES = ("main", "short", "flow", "delta", "relvol", "bos")
rows_cap, rows_unc = [], []
start = 2 * W + 5
for t in range(start, n + 1):
    sub = {k: H[k][:t] for k in keys}
    a = compute_sleeve_multipliers(sub["main"], sub["short"], sub["flow"],
                                   sub["delta"], sub["relvol"], sub["bos"],
                                   window=W, max_multiplier=1.0)
    b = compute_sleeve_multipliers(sub["main"], sub["short"], sub["flow"],
                                   sub["delta"], sub["relvol"], sub["bos"],
                                   window=W, max_multiplier=1e9)
    rows_cap.append([a[s] for s in SLEEVES])
    rows_unc.append([b[s] for s in SLEEVES])

C = np.array(rows_cap)
U = np.array(rows_unc)
T = len(C)
print(f"\nevaluated {T} historical dates\n")

print("=" * 72)
print("Q1: DOES THE CAP EVER BIND?")
print("=" * 72)
print(f"{'sleeve':>8s} {'days capped':>12s} {'% days':>8s} "
      f"{'mean uncapped':>14s} {'max uncapped':>13s} {'mean shortfall':>15s}")
any_bind = False
for i, s in enumerate(SLEEVES):
    diff = U[:, i] - C[:, i]
    bind = diff > 1e-9
    nb = int(bind.sum())
    if nb:
        any_bind = True
    short = diff[bind].mean() if nb else 0.0
    print(f"{s:>8s} {nb:12d} {nb/T:7.2%} {U[:,i].mean():14.4f} "
          f"{U[:,i].max():13.4f} {short:15.4f}")

print("\n" + "=" * 72)
print("Q2: EFFECT ON TOTAL BOOK SIZE")
print("=" * 72)
gc, gu = C.sum(axis=1), U.sum(axis=1)
print(f"  gross multiplier, capped   : mean {gc.mean():.4f}  max {gc.max():.4f}")
print(f"  gross multiplier, uncapped : mean {gu.mean():.4f}  max {gu.max():.4f}")
if gu.mean() > 0:
    print(f"  cap removes on average     : {(1 - gc.mean()/gu.mean()):.2%} of gross")
print(f"  worst single day reduction : {(gu - gc).max():.4f} "
      f"({((gu-gc)/np.where(gu>0,gu,np.nan)).max():.2%} of that day's gross)")

print("\n" + "=" * 72)
print("Q3: HOW CLOSE TO THE CAP DOES IT GET? (headroom)")
print("=" * 72)
print("  uncapped pre-clip multiplier percentiles per sleeve:")
for i, s in enumerate(SLEEVES):
    q = np.percentile(U[:, i], [50, 90, 99, 100])
    print(f"    {s:>8s}  p50={q[0]:.4f}  p90={q[1]:.4f}  p99={q[2]:.4f}  max={q[3]:.4f}")

print("\nNOTE: these are the POST-wB/extra_factor values the sizer returns. The")
print("clip is applied to the pre-scale (1/3)/vol_product term, so a sleeve can")
print("show a small final multiplier and still have been clipped upstream --")
print("which is exactly what the 'days capped' column above measures.")

np.save("/tmp/cap_capped.npy", C)
np.save("/tmp/cap_uncapped.npy", U)
