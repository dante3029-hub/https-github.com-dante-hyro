"""
Strategy B - BOS sleeve. Per-coin market-structure short, 4h bars, event-driven
(continuation short in a confirmed downtrend on a fresh swing-low break,
confirmed by negative taker delta). Universe: 27 coins (clean_panel).
Max 6 concurrent, 3-ATR take-profit / 2-ATR stop / close-above-last-swing-high
exit, 30-bar max hold. This IS the validated sleeve from reference_impl.py
(Sharpe 0.26 -> 1.00 driven mostly by the TP distance) -- not a reconstruction.

`run_bos()` in reference_impl.py is written as a full historical backtest
(position queue, max-concurrent gating, daily P&L aggregation) and isn't
directly callable for "is there a fresh signal right now". This module
extracts the exact same regime/trigger/confirm state machine -- same helper
functions (_atr, _pivots, _load_4h), same constants -- and walks it forward
to the latest confirmed bar to answer that question per coin, leaving
position-count gating (max 6 concurrent) and exit management to Phase 3/4
where live open-position state actually lives.
"""
import numpy as np
from reference_impl import (
    _atr, _pivots, _load_4h,
    BOS_PIVOT_K, BOS_ATR_STOP, BOS_TP_ATR,
)


def latest_signal(coin: str):
    """
    Returns None if no data / not enough history, otherwise a dict:
        {triggered: bool, entry_ref: float, stop: float, tp: float, atr: float,
         state: int, i: int}
    triggered=True means: at the latest CONFIRMED 4h bar, this coin is in a
    downtrend (state == -1, from confirmed swing pivots) AND price just broke
    below the last confirmed swing low (fresh_break) AND taker delta at that
    bar was negative (dn < 0) -- the exact regime+trigger+confirm conditions
    in reference_impl.run_bos(), evaluated at the most recent bar instead of
    over full history.

    entry_ref / stop / tp are computed EXACTLY as run_bos() computes them at
    trigger time: entry_ref = o[i+1] (next bar's open -- this bar hasn't
    opened yet for a live system, so treat this as the reference price to
    place the order at, not a guaranteed fill), stop = entry_ref + 2*ATR,
    tp = entry_ref - 3*ATR. An earlier draft of this function incorrectly
    based stop/tp on the trigger bar's close (cl[i]) instead of the actual
    entry price (o[i+1]) -- fixed here to match run_bos() exactly.
    stop/tp/atr are None when ATR is unavailable or non-positive (matches
    run_bos()'s `not np.isnan(A[i]) and A[i] > 0` trade-taken guard).

    DELIBERATE DIVERGENCE FROM run_bos(), flagged explicitly: run_bos() skips
    ahead (`i = j+1`) past bars covered by an already-open simulated trade,
    so it never updates lastH/lastL/prevH/prevL/state using pivots that occur
    during an open trade. This function walks every bar unconditionally and
    always updates the state machine, because a live system needs the true,
    trade-independent trend state at all times -- whether to suppress a fresh
    trigger because we're already short that coin is a position-management
    decision that belongs to Phase 3/4, not to signal generation. For coins
    with frequent trades this CAN produce a different `state`/lastH/lastL at
    a given bar than the backtest's position-blind loop would show; this is
    intentional, not a parity bug, but is not covered by parity_test.py for
    that reason -- BOS parity there is checked only on the trigger condition
    itself while flat (no divergence possible when there's no open trade to
    skip past).
    """
    d4 = _load_4h(coin)
    if d4 is None:
        return None
    ts, o, hh, ll, cl, vv, tb = d4
    n = len(cl)
    if n < BOS_PIVOT_K + 21:
        return None
    A = _atr(hh, ll, cl)
    ph, pl = _pivots(hh, ll, BOS_PIVOT_K)
    dn = np.divide(2 * tb - vv, np.where(vv > 0, vv, np.nan))

    lastH = lastL = prevH = prevL = np.nan
    state = 0
    i = BOS_PIVOT_K + 20
    last_valid = None
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
        fresh_break = (not np.isnan(lastL)) and cl[i] < lastL and cl[i - 1] >= lastL
        trig = fresh_break and state == -1
        trig = trig and (not np.isnan(dn[i])) and dn[i] < 0

        entry_ref = float(o[i + 1])
        has_atr = (not np.isnan(A[i])) and A[i] > 0
        stop = float(entry_ref + BOS_ATR_STOP * A[i]) if has_atr else None
        tp = float(entry_ref - BOS_TP_ATR * A[i]) if has_atr else None
        # run_bos() only actually takes the trade if has_atr is also true --
        # mirror that here so `triggered` means "would run_bos() take this".
        trig = trig and has_atr

        last_valid = dict(
            i=i, triggered=bool(trig),
            entry_ref=entry_ref,
            stop=stop, tp=tp,
            atr=float(A[i]) if not np.isnan(A[i]) else None,
            state=state,
        )
        i += 1
    return last_valid
