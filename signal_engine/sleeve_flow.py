"""
Strategy A - Flow sleeve. Taker-delta divergence vs. cross-sectional baseline,
ensemble of 3 lookbacks (2d, 3d, 5d), 5 long / 5 short, weekly rebalance.
Universe: 25 coins (run/hist).

IMPORTANT -- the ensemble is over P&L, not over raw score. The backtest
(option1_reference.py __main__) runs each lookback as its OWN independent
run_sleeve backtest -- its own top-5/bottom-5 ranking and its own weight
vector -- and only averages the resulting PNL series across the 3 lookbacks:

    Fs = [run_sleeve(sleeve_F(DN, BF, lb), ...) for lb in FLOW_LOOKBACKS]
    F_ens = mean(Fs[0], Fs[1], Fs[2])   # mean of PNL series, not of scores

Because each lookback's pnl_t = w_t^(lb) . R_t is linear in that lookback's
own weight vector, averaging PNL across lookbacks is mathematically the same
as averaging the three INDEPENDENTLY-RANKED weight vectors elementwise --
that is what this module does. Averaging the raw SCORES first and ranking
once (which was wrong in an earlier draft of this file) is NOT equivalent,
since a coin ranked top-5 under one lookback may not be under another.
"""
import numpy as np
from option1_reference import sleeve_F, N_PER_SIDE, FLOW_LOOKBACKS
from .rank_weights import rank_to_weights


def latest_target_weights(DN: np.ndarray, BF: np.ndarray, t: int = None, n: int = N_PER_SIDE):
    """
    DN, BF: universe-A matrices from data_loader.load_universe_a().
    t: bar index to score at (defaults to the last available bar).
    Returns (weights, ok) -- ok is True if AT LEAST ONE lookback had enough
    valid names to rank.

    KNOWN DIVERGENCE FROM THE BACKTEST (flagged, not silently papered over):
    if a lookback can't rank on a given day (fewer than 2n+2 valid names),
    run_sleeve() in the backtest carries that lookback's PREVIOUS weight
    vector forward into the ensemble average. This stateless snapshot
    function has no memory of prior weights, so it instead contributes an
    all-zero vector for that lookback on such a day. This should be rare with
    full live coverage (it requires >13 of 25 coins missing taker-delta data
    for that lookback window) but is not proven identical -- Phase 3's
    stateful portfolio layer should carry forward the last valid weight per
    lookback if this needs to be closed exactly.
    """
    if t is None:
        t = DN.shape[0] - 1
    weights = []
    any_ok = False
    for lb in FLOW_LOOKBACKS:
        sig_fn = sleeve_F(DN, BF, lb)
        s = sig_fn(t)
        w_lb, ok_lb = rank_to_weights(s, n)
        weights.append(w_lb)
        any_ok = any_ok or ok_lb
    w = np.mean(np.vstack(weights), axis=0)
    return w, any_ok
