"""
Strategy B - DELTA sleeve. Cross-sectional taker-delta ranking (3d lookback),
6 long / 6 short, weekly rebalance. Universe: 27 coins (clean_panel).

`sig_delta` is imported unchanged from reference_impl.py.
"""
import numpy as np
from reference_impl import sig_delta, N_PER_SIDE
from .rank_weights import rank_to_weights


def latest_target_weights(DN: np.ndarray, t: int = None, n: int = N_PER_SIDE):
    """
    DN: universe-B taker-delta matrix from data_loader.load_universe_b().
    t: bar index to score at (defaults to the last available bar).
    Returns (weights, ok) -- see sleeve_main.latest_target_weights for shape.
    """
    if t is None:
        t = DN.shape[0] - 1
    sig_fn = sig_delta(DN)
    s = sig_fn(t)
    w, ok = rank_to_weights(s, n)
    return w, ok
