"""
Within-strategy sleeve combination: turns 3 sleeve return series (Strategy A:
Main/Short/Flow) or 3 sleeve return series (Strategy B: DELTA/RELVOL/BOS)
into ONE combined return series per strategy, matching the exact weights
used to build the backtested `combo_a_correct.npy` / `combo_b_correct.npy`
series that everything downstream (dynamic_weight_test.py, every Monte Carlo
run) was built from.

All functions here operate on 1D daily-return arrays (fraction of $200k
reference notional per day), aligned on a common date index by the caller.
"""
import numpy as np


def strategy_a_combo(main_ret: np.ndarray, short_ret: np.ndarray, flow_ret: np.ndarray) -> np.ndarray:
    """
    eq_thirds = (main + short + flow) / 3 -- EXACT match to build_blend.py's
    `eq_thirds_ret = (m + s + f) / 3.0`. Equal, un-fit weights -- the
    STRATEGY.md finding was explicit that walk-forward-optimized weights
    should NOT be used ("Do not optimize these weights. If you use any
    blend, use naive equal-risk"). No normalization step, so there is no
    causality concern here: this is bit-identical to what was backtested,
    given the same three input return series.
    """
    return (main_ret + short_ret + flow_ret) / 3.0


def _trailing_nz(x: np.ndarray, window: int) -> np.ndarray:
    """
    Causal (trailing-window) analog of reference_impl.py's `nz = lambda x:
    x / x.std() if x.std() > 0 else x`.

    FLAGGED DEVIATION: reference_impl.py's own nz() divides by the WHOLE
    SAMPLE's std (a single number computed once, using the entire history --
    future included). That is look-ahead and cannot be reproduced by a live
    system, which only ever has trailing data. This function instead divides
    x[t] by the trailing `window`-day std of x up to (not including) t,
    matching the same causal-windowing convention already used and validated
    for the Strategy A/B blend in dynamic_weight_test.py's inv_vol_weighted().
    Before `window` days of history exist, returns x[t] unscaled (divisor 1.0).

    This means: the printed Strategy B Sharpe (2.09 core / 2.38 blend) in
    reference_impl.py's docstring was measured under the LOOK-AHEAD nz(), not
    this causal version. Using this function live is a legitimate, necessary
    substitution for real-time trading, but it is a NEW, UNVALIDATED
    normalization -- it has not been separately backtested, and there is no
    guarantee it reproduces anything close to 2.09/2.38. This should be
    backtested on history (using ONLY trailing windows, never full-sample)
    before being trusted, not assumed equivalent by construction.
    """
    n = len(x)
    out = np.array(x, dtype=float, copy=True)
    for t in range(n):
        if t < window:
            continue  # not enough trailing history -- leave unscaled
        s = x[t - window:t].std()
        if s > 1e-12:
            out[t] = x[t] / s
    return out


# Reverse-engineered from /tmp/combo_b_correct.npy (regression fit on the exact
# calendar-aligned 568-day window [2025-01-02, 2026-07-23]: corr=0.986, R^2=0.973
# -- see backtest_causal_b_v2.py). reference_impl.py's nz()-based blend is
# DIMENSIONLESS (each nz() divides by a std, so the raw blend has ~1500% ann
# vol and cannot be sized against dollar notional as-is). The actual cached
# combo_b_correct.npy -- used everywhere downstream (dynamic_weight_test.py,
# every Monte Carlo run in STRATEGY.md) -- is that dimensionless blend RESCALED
# to a target daily vol of ~0.963%/day (~18.4% ann, close to DELTA's own
# ~16.9% ann and RELVOL's own ~17.9% ann vol). THIS RESCALE STEP WAS NOT
# DOCUMENTED ANYWHERE IN THE WORKSPACE'S CODE before this Phase 3 backtest --
# it is a genuinely new finding, not previously surfaced. R^2=0.973, NOT 1.0:
# even under the exact look-ahead nz() and this rescale, my reconstruction
# does not exactly reproduce combo_b_correct.npy (Sharpe of the reconstruction
# on the correct window is 2.59 vs the real cached series' 2.09 -- a ~24%
# relative inflation). This means combo_b_correct.npy embeds at least one
# more undisclosed difference (most likely a different/frozen data snapshot,
# since the live clean_panel has grown ~13 days since that file was cached,
# or a slightly different universe selection) that could NOT be tracked down
# in this pass. Treat TARGET_DAILY_VOL as an empirically fitted number to be
# reviewed, not a backtested design parameter.
TARGET_DAILY_VOL = 0.00963175


def strategy_b_combo(delta_ret: np.ndarray, relvol_ret: np.ndarray, bos_ret: np.ndarray,
                      window: int = 30, W_CORE: float = 2 / 3, W_BOS: float = 1 / 3,
                      target_daily_vol: float = TARGET_DAILY_VOL) -> np.ndarray:
    """
    core = (nz(delta) + nz(relvol)) / 2
    blend_raw = W_CORE * nz(core) + W_BOS * nz(bos)
    blend = (blend_raw / trailing_std(blend_raw, window)) * target_daily_vol   [vol-targeting rescale, see module-level note above]

    Same structure and same W_CORE/W_BOS=2/3, 1/3 weights as
    reference_impl.py's main(), but with _trailing_nz() (causal) substituted
    for nz() (look-ahead) at every normalization step -- see _trailing_nz's
    docstring for why, and for the explicit caveat that this makes
    strategy_b_combo a new, not-yet-independently-validated construction,
    not a proven-identical live replica of the backtested Strategy B number.
    The final vol-targeting rescale is ALSO causal (trailing window, not
    whole-sample) for the same live-executability reason.

    Before `window` days of history, returns 0.0 for that position rather than
    guessing a vol-normalized value from insufficient data -- do not trade
    Strategy B live until `window` days of trailing history exist.

    default window=30 reuses the same trailing window already fit-and-
    validated (via walk-forward: fit on 1st half, unchanged on 2nd half) for
    the Strategy A/B blend in ab_blend.py -- chosen for consistency, not
    because 30 was separately re-optimized for this internal B combination.
    """
    core = (_trailing_nz(delta_ret, window) + _trailing_nz(relvol_ret, window)) / 2
    blend_raw = W_CORE * _trailing_nz(core, window) + W_BOS * _trailing_nz(bos_ret, window)
    n = len(blend_raw)
    blend = np.zeros(n)
    for t in range(n):
        if t < window:
            continue
        trailing_std = blend_raw[t - window:t].std()
        if trailing_std > 1e-12:
            blend[t] = (blend_raw[t] / trailing_std) * target_daily_vol
    return blend
