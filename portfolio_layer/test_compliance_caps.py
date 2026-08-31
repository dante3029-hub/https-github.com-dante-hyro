"""
Regression tests for the two HyroTrader compliance caps added 2026-08-10:
  1. Max-loss-per-trade ($6,000 = 3% of $200k), enforced as a per-leg
     NOTIONAL cap (never a stop-fraction change).
  2. Low-cap exposure cap ($10,000 = 5% of $200k), enforced as an aggregate
     scale-down across all sleeves/coins in the low-cap set.

Each behavioral test is paired with a negative control that proves the test
would actually fail if the cap were silently disabled/broken -- this
codebase was previously caught shipping vacuous tests (see
STOP_SYNC_AND_SIZING_FINDINGS.md), so every assertion here is checked
against a deliberately-uncapped run to confirm it has teeth.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from portfolio_layer.portfolio import (
    size_portfolio, SLEEVE_NAMES, REFERENCE_NOTIONAL,
    DEFAULT_STOP_FRAC, MAX_LOSS_PER_TRADE, LOW_CAP_EXPOSURE_LIMIT,
)
from portfolio_layer.risk_overlay import RiskState


def _flat_risk_state(equity=200_000.0):
    rs = RiskState.new(starting_equity=equity)
    rs.start_new_session(session_date=__import__("datetime").date(2026, 8, 10))
    return rs


def _big_history(n=90, val=0.0):
    return np.full(n, val)


def _varied_history(n=90, seed=0, scale=0.01):
    rng = np.random.RandomState(seed)
    return rng.normal(loc=0.0, scale=scale, size=n)


def _make_weights(main_weight=1.0, coin="BIGCOIN"):
    """A single, oversized leg in the 'main' sleeve at weight=1.0 (i.e. the
    ENTIRE $200k reference notional on one coin) -- deliberately unrealistic
    vs the real 0.10-per-leg backtested construction, specifically so the
    notional cap has clear room to bind and the test is not vacuously
    passing just because normal position sizes never approach $10,000."""
    w = {name: {} for name in SLEEVE_NAMES}
    w["main"][coin] = main_weight
    return w


def test_max_loss_cap_binds_and_clips_to_exact_notional():
    """A main-sleeve leg sized at the full $200k reference notional (weight=1.0,
    multiplier forced to 1.0 via max_sleeve_multiplier + L=1.0) must be clipped
    to MAX_LOSS_PER_TRADE / DEFAULT_STOP_FRAC = $6,000 / 0.60 = $10,000."""
    rs = _flat_risk_state()
    weights = _make_weights(main_weight=1.0, coin="BIGCOIN")
    hist = {name: _big_history() for name in SLEEVE_NAMES}

    sized = size_portfolio(
        sleeve_raw_weights=weights,
        sleeve_return_history=hist,
        risk_state=rs,
        L=1.0,
        max_sleeve_multiplier=1e9,  # let main's multiplier reach wA*(1/3) unclipped by the vol-safety cap
    )
    # main sleeve multiplier with all-zero history collapses to wA*(1/3); to make
    # this test assert an EXACT, hand-computable number regardless of that
    # internal formula, size with max_loss_per_trade turned off first to see the
    # true uncapped notional, then confirm the capped run matches the formula.
    sized_uncapped = size_portfolio(
        sleeve_raw_weights=weights, sleeve_return_history=hist, risk_state=rs,
        L=1.0, max_sleeve_multiplier=1e9, max_loss_per_trade=0,
    )
    uncapped_d = sized_uncapped.dollar_targets["main"]["BIGCOIN"]
    expected_cap = MAX_LOSS_PER_TRADE / DEFAULT_STOP_FRAC
    assert abs(uncapped_d) > expected_cap, (
        "test setup is vacuous: uncapped leg is not even bigger than the cap, "
        f"got uncapped=${uncapped_d:,.2f} vs cap=${expected_cap:,.2f}"
    )

    capped_d = sized.dollar_targets["main"]["BIGCOIN"]
    assert abs(capped_d) == expected_cap, f"expected exactly ${expected_cap:,.2f}, got ${capped_d:,.2f}"
    assert any("max-loss-per-trade cap" in f for f in sized.flags), "cap must be flagged in the audit trail"


def test_max_loss_cap_does_not_touch_small_legs():
    """Negative control: a leg already well under the cap must pass through
    UNCHANGED (proves the cap is a clip, not a blanket rescale)."""
    rs = _flat_risk_state()
    weights = _make_weights(main_weight=0.01, coin="SMALLCOIN")  # tiny weight -> tiny notional
    hist = {name: _big_history() for name in SLEEVE_NAMES}
    sized = size_portfolio(sleeve_raw_weights=weights, sleeve_return_history=hist,
                            risk_state=rs, L=1.0, max_sleeve_multiplier=1e9)
    d = sized.dollar_targets["main"]["SMALLCOIN"]
    assert abs(d) < MAX_LOSS_PER_TRADE / DEFAULT_STOP_FRAC
    assert not any("max-loss-per-trade cap" in f for f in sized.flags)


def test_max_loss_cap_disabled_reproduces_uncapped_notional():
    """Negative control: max_loss_per_trade=0 must exactly reproduce the
    pre-existing (pre-this-change) sizing formula -- proves the cap is
    opt-in/backward compatible, not silently always-on."""
    rs = _flat_risk_state()
    weights = _make_weights(main_weight=1.0, coin="BIGCOIN")
    hist = {name: _big_history() for name in SLEEVE_NAMES}
    sized_off = size_portfolio(sleeve_raw_weights=weights, sleeve_return_history=hist,
                                risk_state=rs, L=1.0, max_sleeve_multiplier=1e9, max_loss_per_trade=0)
    d = sized_off.dollar_targets["main"]["BIGCOIN"]
    expected_cap = MAX_LOSS_PER_TRADE / DEFAULT_STOP_FRAC
    assert abs(d) > expected_cap, "cap=0 must NOT clip -- if this fails the disable switch is broken"


def test_max_loss_cap_respects_custom_stop_frac_for_event_sleeves():
    """A short/bos-style tighter ATR stop_frac must produce a LOOSER dollar cap
    than the 0.60 default -- confirms the per-sleeve/per-coin override path
    (used for event sleeves' real ATR stop) actually changes the outcome, not
    just accepted-and-ignored.

    2026-08-10 UPDATED. Original assertion expected $30,000 for a 0.20 stop
    (6000/0.20). MAX_NOTIONAL_PER_LEG (gap cap, $10,000) now clamps that: a
    $30,000 leg that GAPS 60% loses $18,000, breaching the $6,000 per-trade
    rule that no stop can prevent. The override path is therefore exercised
    here at stop fracs where the gap cap is NOT the binding constraint, so the
    test still proves what it was written to prove."""
    rs = _flat_risk_state()
    weights = {name: {} for name in SLEEVE_NAMES}
    weights["short"]["ALTCOIN"] = -1.0  # oversized on purpose
    hist = {name: _big_history() for name in SLEEVE_NAMES}

    # 1.20 and 0.60: both give stop-out caps ($5,000 and $10,000) at or below
    # the $10,000 gap cap, so the override path is the binding constraint here.
    tight = size_portfolio(sleeve_raw_weights=weights, sleeve_return_history=hist, risk_state=rs,
                            L=1.0, max_sleeve_multiplier=1e9,
                            stop_fracs={"short": {"ALTCOIN": 1.20}})
    loose_default = size_portfolio(sleeve_raw_weights=weights, sleeve_return_history=hist, risk_state=rs,
                                    L=1.0, max_sleeve_multiplier=1e9)

    d_tight = abs(tight.dollar_targets["short"]["ALTCOIN"])
    d_default = abs(loose_default.dollar_targets["short"]["ALTCOIN"])
    assert abs(d_tight - 6_000.0 / 1.20) < 1e-6, f"expected $5,000 cap, got ${d_tight:,.2f}"
    assert abs(d_default - 6_000.0 / DEFAULT_STOP_FRAC) < 1e-6, f"expected $10,000 cap, got ${d_default:,.2f}"
    assert d_tight < d_default, "a WIDER stop_frac must produce a TIGHTER notional cap"


def test_gap_cap_binds_when_stop_cap_would_allow_a_rule_breach():
    """2026-08-10 NEW. The per-trade cap protects against a STOP-OUT. It does not
    protect against price GAPPING THROUGH the stop. Worst adverse single-day move
    in the traded universe was TRX +95.8%.

    With a 0.20 stop the stop-out cap alone would allow $30,000 of notional; a
    60% gap on that is $18,000 -- 3x the $6,000 per-trade limit. MAX_NOTIONAL_PER_LEG
    must clamp it."""
    from portfolio_layer.portfolio import MAX_NOTIONAL_PER_LEG, ASSUMED_ADVERSE_GAP, MAX_LOSS_PER_TRADE
    rs = _flat_risk_state()
    weights = {name: {} for name in SLEEVE_NAMES}
    weights["short"]["ALTCOIN"] = -1.0
    hist = {name: _big_history() for name in SLEEVE_NAMES}
    sized = size_portfolio(sleeve_raw_weights=weights, sleeve_return_history=hist, risk_state=rs,
                            L=1.0, max_sleeve_multiplier=1e9,
                            stop_fracs={"short": {"ALTCOIN": 0.20}})
    d = abs(sized.dollar_targets["short"]["ALTCOIN"])
    assert d <= MAX_NOTIONAL_PER_LEG + 1e-6, f"gap cap not binding: ${d:,.2f}"
    assert d * ASSUMED_ADVERSE_GAP <= MAX_LOSS_PER_TRADE + 1e-6, \
        f"a {ASSUMED_ADVERSE_GAP:.0%} gap on ${d:,.2f} breaches ${MAX_LOSS_PER_TRADE:,.0f}"


def test_low_cap_exposure_cap_scales_down_across_sleeves():
    """Two low-cap legs in DIFFERENT sleeves summing above $10,000 must be
    scaled down proportionally so their combined absolute notional is
    exactly the limit."""
    rs = _flat_risk_state()
    weights = {name: {} for name in SLEEVE_NAMES}
    # main and flow share the IDENTICAL multiplier formula (m_main = m_flow =
    # wA * (1/3), see position_sizer.py docstring) so a plain zero/flat
    # history is enough -- no need for delta/relvol/bos's vol-scaled formula.
    weights["main"]["ORDI"] = 0.30
    weights["flow"]["BRETT"] = 0.20
    hist = {name: _big_history() for name in SLEEVE_NAMES}

    sized_nocap = size_portfolio(sleeve_raw_weights=weights, sleeve_return_history=hist, risk_state=rs,
                                  L=1.0, max_sleeve_multiplier=1e9, max_loss_per_trade=0,
                                  low_cap_coins=None)
    ordi_before = abs(sized_nocap.dollar_targets["main"]["ORDI"])
    brett_before = abs(sized_nocap.dollar_targets["flow"]["BRETT"])
    total_before = ordi_before + brett_before
    assert total_before > LOW_CAP_EXPOSURE_LIMIT, (
        f"test setup is vacuous: combined low-cap notional ${total_before:,.2f} must exceed "
        f"the ${LOW_CAP_EXPOSURE_LIMIT:,.0f} limit for this test to prove anything"
    )

    sized_capped = size_portfolio(sleeve_raw_weights=weights, sleeve_return_history=hist, risk_state=rs,
                                   L=1.0, max_sleeve_multiplier=1e9, max_loss_per_trade=0,
                                   low_cap_coins={"ORDI", "BRETT"})
    ordi_after = abs(sized_capped.dollar_targets["main"]["ORDI"])
    brett_after = abs(sized_capped.dollar_targets["flow"]["BRETT"])
    total_after = ordi_after + brett_after

    assert abs(total_after - LOW_CAP_EXPOSURE_LIMIT) < 1e-6, f"expected exactly $10,000, got ${total_after:,.2f}"
    # proportional, not equal-split: ratio between the two legs must be preserved
    assert abs(ordi_after / brett_after - ordi_before / brett_before) < 1e-6
    assert any("low-cap exposure cap" in f for f in sized_capped.flags)


def test_low_cap_exposure_cap_ignores_non_low_cap_coins():
    """Negative control: an oversized leg in a coin NOT in low_cap_coins must
    be left alone by the low-cap cap (proves the cap is coin-set-scoped, not
    a blanket portfolio-wide notional cap in disguise)."""
    rs = _flat_risk_state()
    weights = {name: {} for name in SLEEVE_NAMES}
    weights["main"]["BTC"] = 1.0  # huge leg, but BTC is not low-cap
    hist = {name: _big_history() for name in SLEEVE_NAMES}
    sized = size_portfolio(sleeve_raw_weights=weights, sleeve_return_history=hist, risk_state=rs,
                            L=1.0, max_sleeve_multiplier=1e9, max_loss_per_trade=0,
                            low_cap_coins={"ORDI", "BRETT"})
    d = abs(sized.dollar_targets["main"]["BTC"])
    assert d > LOW_CAP_EXPOSURE_LIMIT, "BTC leg must be untouched by the low-cap cap"
    assert not any("low-cap exposure cap" in f for f in sized.flags)


def test_low_cap_exposure_cap_disabled_by_default_empty_set():
    """Negative control: passing low_cap_coins=None (the size_portfolio
    default) must never trigger the cap, however large the notional --
    confirms callers who don't opt in are unaffected."""
    rs = _flat_risk_state()
    weights = _make_weights(main_weight=1.0, coin="ORDI")
    hist = {name: _big_history() for name in SLEEVE_NAMES}
    sized = size_portfolio(sleeve_raw_weights=weights, sleeve_return_history=hist, risk_state=rs,
                            L=1.0, max_sleeve_multiplier=1e9, max_loss_per_trade=0)
    d = abs(sized.dollar_targets["main"]["ORDI"])
    assert d > LOW_CAP_EXPOSURE_LIMIT
    assert not any("low-cap exposure cap" in f for f in sized.flags)


def test_both_caps_compose_loss_cap_applied_before_low_cap_cap():
    """When both caps are active, the loss cap runs first (per-leg), then the
    low-cap aggregate cap runs on the POST-loss-cap notionals. With a single
    oversized low-cap leg that already exceeds both individually, the final
    notional must be the TIGHTER of the two, i.e. min($10,000 loss-cap,
    $10,000 low-cap-limit) trivially -- so use two low-cap legs where the
    per-leg loss cap binds on one but the aggregate low-cap cap must still
    bind on the (already loss-capped) total.
    """
    rs = _flat_risk_state()
    weights = {name: {} for name in SLEEVE_NAMES}
    # main and flow share the IDENTICAL multiplier formula, so symmetric
    # weights here produce exactly symmetric pre-cap notionals.
    weights["main"]["ORDI"] = 1.0     # will be clipped to $10,000 by the loss cap alone
    weights["flow"]["BRETT"] = 1.0    # will also be clipped to $10,000 by the loss cap alone
    hist = {name: _big_history() for name in SLEEVE_NAMES}

    sized = size_portfolio(sleeve_raw_weights=weights, sleeve_return_history=hist, risk_state=rs,
                            L=1.0, max_sleeve_multiplier=1e9,
                            low_cap_coins={"ORDI", "BRETT"})
    ordi = abs(sized.dollar_targets["main"]["ORDI"])
    brett = abs(sized.dollar_targets["flow"]["BRETT"])
    total = ordi + brett
    # after the per-leg loss cap both legs are $10,000 each ($20,000 combined),
    # which still exceeds the $10,000 aggregate low-cap limit, so the low-cap
    # cap must scale BOTH down further, equally (symmetric legs).
    assert abs(total - LOW_CAP_EXPOSURE_LIMIT) < 1e-6, f"expected $10,000 combined, got ${total:,.2f}"
    assert abs(ordi - brett) < 1e-6, "symmetric legs must be scaled identically"
    assert ordi < MAX_LOSS_PER_TRADE / DEFAULT_STOP_FRAC, (
        "each leg must end up SMALLER than the standalone loss cap once the "
        "aggregate low-cap cap also applies on top"
    )
    assert any("max-loss-per-trade cap" in f for f in sized.flags)
    assert any("low-cap exposure cap" in f for f in sized.flags)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


def test_live_stop_and_aggregate_stop_stay_in_sync():
    """
    2026-08-10 REGRESSION GUARD. AGGREGATE_STOP_FRAC_DEFAULT must track
    bot/config.LIVE_STOP_LOSS_FRAC. They are separate constants in separate
    packages; if one is changed without the other, the aggregate cap silently
    mis-measures crash risk. That exact drift (0.60 vs 0.24) throttled live size
    2.5x harder than the risk warranted.

    DEFAULT_STOP_FRAC is deliberately NOT synced -- it is the conservative
    denominator for the per-trade dollar cap only.
    """
    import os, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.join(root, "bot") not in sys.path:
        sys.path.insert(0, os.path.join(root, "bot"))
    import config as live_cfg
    from portfolio_layer.portfolio import AGGREGATE_STOP_FRAC_DEFAULT, DEFAULT_STOP_FRAC

    assert abs(live_cfg.LIVE_STOP_LOSS_FRAC - AGGREGATE_STOP_FRAC_DEFAULT) < 1e-9, (
        f"drift: LIVE_STOP_LOSS_FRAC={live_cfg.LIVE_STOP_LOSS_FRAC} but "
        f"AGGREGATE_STOP_FRAC_DEFAULT={AGGREGATE_STOP_FRAC_DEFAULT}")
    assert DEFAULT_STOP_FRAC != AGGREGATE_STOP_FRAC_DEFAULT, \
        "the per-trade cap denominator must stay conservative and separate"


def test_live_stop_satisfies_3pct_rule_at_max_leg():
    """At the largest permitted leg, a stop-out must stay under $6,000."""
    import os, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.join(root, "bot") not in sys.path:
        sys.path.insert(0, os.path.join(root, "bot"))
    import config as live_cfg
    from portfolio_layer.portfolio import MAX_NOTIONAL_PER_LEG, MAX_LOSS_PER_TRADE
    loss = live_cfg.LIVE_STOP_LOSS_FRAC * MAX_NOTIONAL_PER_LEG
    assert loss <= MAX_LOSS_PER_TRADE + 1e-6, \
        f"stop-out at max leg = ${loss:,.0f} > ${MAX_LOSS_PER_TRADE:,.0f}"
