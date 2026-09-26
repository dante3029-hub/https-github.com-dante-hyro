"""Pairs / relative value — the only mechanism genuinely orthogonal to the book.

WHY: five of eight sleeves are "long-only price structure on 6-12h bars". They
all answer the same question (is price continuing up?) so they correlate 0.4-0.6
with each other. A pairs trade has NO directional view -- it profits from two
correlated coins converging, whatever the market does.

MECHANISM
  1. find pairs with high trailing correlation (they move together)
  2. build the spread: log(A) - beta*log(B), beta from a rolling regression
  3. z-score the spread on a rolling window
  4. when z > entry: short A, long B (expect convergence). Mirror for z < -entry
  5. exit when z crosses back through `exit_z`, or on a time cap

All formation stats use ONLY trailing data -- correlation, beta and the z-score
mean/sd are computed on the window BEFORE the bar being evaluated.
"""
from __future__ import annotations
import itertools, os
import numpy as np, pandas as pd

TAKER = '/tmp/hyro/taker_data'
FEE = 0.00085


def build_spreads(PX, form_bars=180, min_corr=0.7):
    """Rank pairs by trailing correlation on the FORMATION window only."""
    lp = np.log(PX)
    R = lp.diff()
    form = R.iloc[:form_bars]
    C = form.corr()
    pairs = []
    for a, b in itertools.combinations(PX.columns, 2):
        c = C.loc[a, b]
        if np.isfinite(c) and c >= min_corr:
            pairs.append((a, b, float(c)))
    pairs.sort(key=lambda x: -x[2])
    return pairs


def trade_pair(PX, a, b, lookback=90, entry_z=2.0, exit_z=0.5,
               max_hold=60, stop_z=4.0):
    """Returns (entry_i, exit_i, side_a) -- side_a +1 means long A short B."""
    la, lb = np.log(PX[a].to_numpy(float)), np.log(PX[b].to_numpy(float))
    n = len(la)
    out = []
    pos = 0
    ei = 0
    for i in range(lookback + 2, n - 2):
        # rolling hedge ratio and z-score, trailing window only
        wa, wb = la[i-lookback:i], lb[i-lookback:i]
        if not (np.isfinite(wa).all() and np.isfinite(wb).all()):
            continue
        vb = wb.var()
        if vb <= 0:
            continue
        beta = np.cov(wa, wb)[0, 1] / vb
        sp = wa - beta*wb
        mu, sd = sp.mean(), sp.std()
        if sd <= 0:
            continue
        zt = (la[i] - beta*lb[i] - mu) / sd
        if pos == 0:
            if zt >= entry_z:
                pos, ei = -1, i          # spread too high -> short A, long B
            elif zt <= -entry_z:
                pos, ei = 1, i
        else:
            hit_exit = (pos == -1 and zt <= exit_z) or (pos == 1 and zt >= -exit_z)
            hit_stop = abs(zt) >= stop_z
            timeout = (i - ei) >= max_hold
            if hit_exit or hit_stop or timeout:
                out.append((ei + 1, i, pos))
                pos = 0
    return out
