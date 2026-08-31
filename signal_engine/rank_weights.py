"""
Shared cross-sectional ranking -> target-weight construction.

This is copied VERBATIM (not reimplemented) from the weight-construction lines
inside run_sleeve() in both reference_impl.py and option1_reference.py, which
are identical in both files:

    order = np.argsort(np.where(valid, s, -1e9))
    w = np.zeros(C); w[order[-n:]] = 0.5/n; w[order[:n]] = -0.5/n

Used by: sleeve_main, sleeve_flow (Strategy A) and sleeve_delta, sleeve_relvol
(Strategy B). Kept in one place so all four dense cross-sectional sleeves are
guaranteed to rank identically to what was backtested -- one bug fixed here
fixes it everywhere instead of needing four separate patches.
"""
import numpy as np


def rank_to_weights(s: np.ndarray, n: int):
    """
    s: 1D score array, length C (one score per coin), NaN where score is
       unavailable/invalid for that coin.
    n: number of names per side (long top-n, short bottom-n).

    Returns (w, ok) where w is a length-C weight array (0.5/n on the n
    highest-scoring names, -0.5/n on the n lowest-scoring names, 0 elsewhere)
    and ok is False if there weren't enough valid names to fill both sides
    (mirrors run_sleeve's `if valid.sum() < 2*n + 2: skip` guard).
    """
    s = np.asarray(s, dtype=float)
    C = s.shape[0]
    valid = ~np.isnan(s)
    if valid.sum() < 2 * n + 2:
        return np.zeros(C), False
    order = np.argsort(np.where(valid, s, -1e9))
    w = np.zeros(C)
    w[order[-n:]] = 0.5 / n
    w[order[:n]] = -0.5 / n
    return w, True
