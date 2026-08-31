"""
Between-strategy blend: 30-day trailing inverse-vol weight between Strategy A
(eq_thirds) and Strategy B (2/3 core + 1/3 BOS). This is the
`inv_vol_weighted()` function from dynamic_weight_test.py, copied VERBATIM
(not reimplemented) -- it is already causal in the original backtest (uses
only a[t-W:t], b[t-W:t], excludes day t) so, unlike sleeve_combiner.py's
internal Strategy-B normalization, this step carries NO new deviation from
what was validated: W=30 was chosen by grid search on the first half only
and held UNCHANGED on the second half (walk-forward), and the resulting
`combo_invvol.npy` (Sharpe 2.475, CAGR 42.08%, MaxDD 6.28%) is the exact
series every downstream Monte Carlo run in STRATEGY.md used.
"""
import numpy as np

DEFAULT_WINDOW = 30  # walk-forward-selected: fit on 1st half, held on 2nd half (STRATEGY.md §~dynamic_weight_test)


def inv_vol_weighted(a: np.ndarray, b: np.ndarray, W: int = DEFAULT_WINDOW):
    """
    a: Strategy A combined return series (eq_thirds)
    b: Strategy B combined return series (2/3 core + 1/3 BOS)
    Returns (combo, wA) where wA[t] is the fraction of portfolio risk
    allocated to Strategy A at day t, (1-wA[t]) to Strategy B.

    Verbatim logic from dynamic_weight_test.py's inv_vol_weighted(): before
    W days of history, wA defaults to 0.5 (naive 50/50, not enough data to
    estimate trailing vol yet); afterwards, wA = (1/vol_a) / (1/vol_a + 1/vol_b)
    using the trailing W-day standard deviation of each series, EXCLUDING day t.
    """
    n = len(a)
    wA = np.full(n, 0.5)
    combo = np.zeros(n)
    for t in range(n):
        if t < W:
            wa = 0.5
        else:
            vola = a[t - W:t].std()
            volb = b[t - W:t].std()
            if vola <= 1e-12 or volb <= 1e-12:
                wa = 0.5
            else:
                inv_a, inv_b = 1 / vola, 1 / volb
                wa = inv_a / (inv_a + inv_b)
        wA[t] = wa
        combo[t] = wa * a[t] + (1 - wa) * b[t]
    return combo, wA


def latest_weight(a: np.ndarray, b: np.ndarray, W: int = DEFAULT_WINDOW):
    """
    Live convenience wrapper: given the full trailing history of a and b up
    to and including "today" (a[-1], b[-1] are today's realized returns),
    returns the weight (wA_today, wB_today) to apply to TOMORROW's sleeve
    sizing -- i.e. computed from the trailing W days ENDING today (a[-W:],
    b[-W:]), matching the backtest's `a[t-W:t]` (excludes day t) convention
    applied one day ahead. If fewer than W days of history exist, returns
    (0.5, 0.5).
    """
    n = len(a)
    if n < W:
        return 0.5, 0.5
    vola = a[-W:].std()
    volb = b[-W:].std()
    if vola <= 1e-12 or volb <= 1e-12:
        return 0.5, 0.5
    inv_a, inv_b = 1 / vola, 1 / volb
    wa = inv_a / (inv_a + inv_b)
    return wa, 1 - wa
