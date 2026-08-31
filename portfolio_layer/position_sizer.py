"""
Derives the SCALAR multiplier to apply to each of the 6 sleeves' raw target
weight vectors (from signal_engine, each summing to ~1.0 gross against the
$200k reference notional -- i.e. "100% notional deployed") so that combining
them reproduces the exact nested combination recipe backtested in
STRATEGY.md / reference_impl.py / dynamic_weight_test.py:

    combo_A = (main + short + flow) / 3                              [[build_blend.py]]
    core    = (nz(delta) + nz(relvol)) / 2                           [[reference_impl.py]]
    combo_B = ((2/3)*nz(core) + (1/3)*nz(bos)) rescaled to target vol [[reference_impl.py + reverse-engineered rescale, see sleeve_combiner.py]]
    combo   = wA*combo_A + (1-wA)*combo_B                            [[dynamic_weight_test.py]]

Expanding this out algebraically (all nz() replaced with the causal
_trailing_nz() from sleeve_combiner.py -- see that module's docstring for why
this is a flagged, unvalidated substitution for reference_impl.py's
look-ahead nz()):

    m_main   = wA * (1/3)
    m_short  = wA * (1/3)
    m_flow   = wA * (1/3)
    m_delta  = (1-wA) * (1/3) / (vol_core * vol_delta) * extra_factor
    m_relvol = (1-wA) * (1/3) / (vol_core * vol_relvol) * extra_factor
    m_bos    = (1-wA) / vol_bos * (1/3) * extra_factor

where vol_X is the trailing `window`-day standard deviation of sleeve X's OWN
full-notional daily return history (vol_core is the trailing std of the
already-once-normalized `core` series), and

    extra_factor = target_daily_vol / trailing_std(blend_raw, window)

accounts for the target-vol rescale that sleeve_combiner.strategy_b_combo()
applies at the end (see that module's TARGET_DAILY_VOL note -- reverse
engineered from /tmp/combo_b_correct.npy, R^2=0.973, NOT an exact match).
Derivation of the base terms: core = 0.5*delta/vol_delta + 0.5*relvol/vol_relvol,
so nz(core) = core/vol_core, and blend_raw's DELTA term is
(2/3)*(0.5/vol_core)/vol_delta*delta = (1/3)/(vol_core*vol_delta)*delta;
symmetric for RELVOL. BOS's term is (1/3)*bos/vol_bos directly. The final
blend = (blend_raw / trailing_std(blend_raw)) * target_daily_vol multiplies
every term by the same scalar extra_factor.

SAFETY CAP, NOT PRESENT IN THE BACKTEST: the backtest only ever computed
these multipliers on already-realized historical data, where vol_core /
vol_delta / vol_relvol / vol_bos / vol_blend_raw were never anomalously close
to zero over any real window. A live system faces a real operational risk: a
data glitch, a stale feed, or a genuinely quiet market could make a trailing
vol estimate collapse toward zero, which would make 1/vol blow up and
single-handedly oversize a sleeve to a dangerous multiple of intended
notional. This module adds an explicit `max_multiplier` cap, applied to the
(1/3)/vol_product term BEFORE the wB scale-down and extra_factor, defaulting
to 1.0 -- i.e. no single sleeve's internal vol-scaling can push its
un-blended share above 100% of a naive equal-thirds allocation. This cap is a
deliberately conservative placeholder, NOT a backtested parameter -- the
original backtest never needed one because it only ever measured realized
historical vol, never risked a live divide-by-near-zero. Review and tune
before live use.
"""
import numpy as np

from .sleeve_combiner import strategy_a_combo, strategy_b_combo, _trailing_nz, TARGET_DAILY_VOL
from .ab_blend import latest_weight, DEFAULT_WINDOW

# 2026-08-12: Strategy A DISABLED.
# option1_reference.build_matrices() has NO coin-selection logic -- it globs
# every *_1h.csv in run/hist and drops BTC/ETHBTC. The backtest ran on a dir
# holding 25 files; run/hist now points at a 99-coin panel, so Strategy A would
# trade a universe that was never backtested, and would change silently whenever
# a file is added or removed. Universe B (select_universe(): $5M liquidity
# screen + START date + hard coverage assertion) does not have this defect.
# Re-enable only after Universe A's coin list is pinned explicitly.
STRATEGY_A_ENABLED = False


def _trailing_vol(x: np.ndarray, window: int) -> float:
    if len(x) < window:
        return 0.0
    return float(x[-window:].std())


def compute_sleeve_multipliers(main_hist: np.ndarray, short_hist: np.ndarray, flow_hist: np.ndarray,
                                delta_hist: np.ndarray, relvol_hist: np.ndarray, bos_hist: np.ndarray,
                                window: int = DEFAULT_WINDOW, max_multiplier: float = 1.0,
                                target_daily_vol: float = TARGET_DAILY_VOL) -> dict:
    """
    All *_hist arrays: aligned trailing daily-return history (fraction of
    $200k notional) for each sleeve, assuming that sleeve's raw target weight
    vector was run at full (~1.0 gross) notional every day. Length must be
    >= 2*window for the vol-scaled sleeves (DELTA/RELVOL/BOS) to get a
    non-zero multiplier -- one `window` to build the trailing-vol estimates
    of the raw legs, and another `window` on top of that for the trailing-vol
    estimate of blend_raw itself (extra_factor). Shorter histories default to
    0 (safer than an undefined/guessed scale -- caller should not trade those
    sleeves live until enough history exists).

    Returns a dict: {"main": m, "short": m, "flow": m, "delta": m,
                      "relvol": m, "bos": m, "wA": wA, "wB": wB}
    Each m is the scalar to multiply that sleeve's raw target weight vector
    by (before the account-level L * throttle * kill-switch overlay in
    risk_overlay.py, which applies uniformly across all sleeves afterward).
    """
    combo_a = strategy_a_combo(main_hist, short_hist, flow_hist)
    combo_b = strategy_b_combo(delta_hist, relvol_hist, bos_hist, window=window, target_daily_vol=target_daily_vol)
    wA, wB = latest_weight(combo_a, combo_b, W=window)
    if not STRATEGY_A_ENABLED:
        wA, wB = 0.0, 1.0

    core_hist = (_trailing_nz(delta_hist, window) + _trailing_nz(relvol_hist, window)) / 2
    vol_core = _trailing_vol(core_hist, window)
    vol_delta = _trailing_vol(delta_hist, window)
    vol_relvol = _trailing_vol(relvol_hist, window)
    vol_bos = _trailing_vol(bos_hist, window)

    blend_raw_hist = (2 / 3) * _trailing_nz(core_hist, window) + (1 / 3) * _trailing_nz(bos_hist, window)
    vol_blend_raw = _trailing_vol(blend_raw_hist, window)
    extra_factor = (target_daily_vol / vol_blend_raw) if vol_blend_raw > 1e-12 else 0.0

    def scaled(denom_product, static_share):
        if denom_product <= 1e-12:
            return 0.0
        m = static_share / denom_product
        return float(np.clip(m, 0.0, max_multiplier))

    m_delta = scaled(vol_core * vol_delta, (1 / 3)) * wB * extra_factor
    m_relvol = scaled(vol_core * vol_relvol, (1 / 3)) * wB * extra_factor
    m_bos = scaled(vol_bos, (1 / 3)) * wB * extra_factor

    m_main = wA * (1 / 3)
    m_short = wA * (1 / 3)
    m_flow = wA * (1 / 3)

    return dict(main=m_main, short=m_short, flow=m_flow,
                delta=m_delta, relvol=m_relvol, bos=m_bos,
                wA=wA, wB=wB, extra_factor=extra_factor)
