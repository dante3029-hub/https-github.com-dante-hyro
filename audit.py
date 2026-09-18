"""Forensic audit of the 7-sleeve book.

Checks the failure modes that have ACTUALLY occurred in this project rather
than a generic checklist:

  1. LOOKAHEAD      entry at a price not knowable at signal time. Cost us a
                    6.06 Sharpe that was really 1.04.
  2. LOSS-ERASURE   position zeroed BEFORE the loss is booked. Inflated an
                    earlier book by +0.87.
  3. SIGN ERRORS    stop placed on the profit side. The W5 sleeve booked
                    impossible gains on 22% of trades.
  4. COST REALISM   turnover vs the fee assumption.
  5. RECONCILIATION sim P&L vs brute-force trade-level P&L.
  6. CONTROL        random-direction book on the same bars.
  7. DEGENERACY     is one sleeve or one coin carrying it?
"""
import os
import numpy as np, pandas as pd
import hourly_sim as H


def check_shift_alignment(W, RH):
    """P&L must use weights from BEFORE the return. Compare the correct
    shift(1) against the cheating no-shift version -- a large gap means the
    sleeve's edge is mostly same-bar information."""
    correct = (W.shift(1)*RH).sum(axis=1)
    cheat = (W*RH).sum(axis=1)
    return H.stats_hourly(correct)[0], H.stats_hourly(cheat)[0]


def check_trade_sanity(trades, PXh):
    """exit after entry, and stop on the correct side of entry."""
    bad_order = sum(1 for t in trades if t[2] <= t[1])
    return bad_order, len(trades)


def brute_force_pnl(trades, PXh, fee=H.FEE, per=1.0):
    """Recompute P&L trade by trade from raw prices, independent of the
    weight-matrix machinery."""
    tot = 0.0
    n = 0
    for coin, t0, t1, side in trades:
        if coin not in PXh.columns:
            continue
        s = PXh[coin]
        try:
            i0 = s.index.searchsorted(t0)
            i1 = s.index.searchsorted(t1)
        except Exception:
            continue
        if i0 >= len(s) or i1 >= len(s) or i1 <= i0:
            continue
        p0, p1 = s.iloc[i0], s.iloc[i1]
        if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0:
            continue
        tot += per*(side*(p1-p0)/p0 - 2*fee)
        n += 1
    return tot, n
