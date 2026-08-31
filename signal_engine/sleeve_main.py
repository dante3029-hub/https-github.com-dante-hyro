"""
Strategy A - Main sleeve. Momentum(14d) + funding-carry(7d) cross-sectional
rank, 5 long / 5 short, weekly rebalance. Universe: 25 coins (run/hist).

Signal function `sleeve_A` is imported unchanged from option1_reference.py --
not reimplemented -- so live scoring is guaranteed identical to the backtest
for the same inputs.
"""
import numpy as np
from option1_reference import sleeve_A, N_PER_SIDE
from .rank_weights import rank_to_weights


def latest_target_weights(PX: np.ndarray, FN: np.ndarray, t: int = None, n: int = N_PER_SIDE):
    """
    PX, FN: universe-A matrices from data_loader.load_universe_a().
    t: bar index to score at (defaults to the last available bar, i.e. "today").
    Returns (weights, coin_order_is_caller's_responsibility, ok).
    weights: length-C array, 0.5/n on the n coins with the highest score
             (long), -0.5/n on the n lowest (short), 0 elsewhere.
    ok: False if there weren't enough valid names to fill both sides at t.
    """
    if t is None:
        t = PX.shape[0] - 1
    sig_fn = sleeve_A(PX, FN)
    s = sig_fn(t)
    w, ok = rank_to_weights(s, n)
    return w, ok
