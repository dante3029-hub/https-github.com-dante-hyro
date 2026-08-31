"""
Functional test of compliance.py against both drawdown_type settings
(STATIC -- this account's confirmed real terms as of 2026-08-10 -- and
TRAILING, kept as regression coverage in case the account is ever
downgraded), using concrete numbers to make the mechanics tangible instead
of only described in prose.
"""
import datetime as dt
from compliance import HyroTraderComplianceConfig, ComplianceState, DrawdownFloorMode, DrawdownType


def run_scenario(mode: DrawdownFloorMode, drawdown_type: DrawdownType):
    cfg = HyroTraderComplianceConfig(drawdown_floor_mode=mode, drawdown_type=drawdown_type)
    cs = ComplianceState.start_phase1(cfg)
    cs.start_new_day(dt.date(2026, 1, 1))

    print(f"\n=== drawdown_type={drawdown_type.value} floor_mode={mode.value} ===")
    print(f"Phase 1 start: equity=${cs.equity:,.0f}  max_dd_floor=${cs.dd_floor_dollars:,.0f}")

    # Phase 1: equity runs up to a $232,000 peak (16% gain -- passes the 10%
    # target of $220,000 along the way), then gives back some before Phase 1
    # officially ends at $221,000.
    for eq in (205_000, 215_000, 232_000, 221_000):
        status = cs.update_equity(eq)
    print(f"Phase 1 end:   equity=${cs.equity:,.0f}  peak=${cs.peak_equity:,.0f}  "
          f"max_dd_floor=${cs.dd_floor_dollars:,.0f}  busted={status['busted']}")
    assert not status["busted"], "Phase 1 should not have busted in this scenario"

    cs.advance_to_phase2()
    cs.start_new_day(dt.date(2026, 2, 1))
    print(f"Phase 2 start: equity=${cs.equity:,.0f}  peak=${cs.peak_equity:,.0f}  "
          f"max_dd_floor=${cs.dd_floor_dollars:,.0f}")

    # Phase 2 sees a normal early drawdown down to $206,000 (a 3% dip from the
    # $212k start -- unremarkable, well within the strategy's own historical
    # ~7-10% max drawdowns per Phase 3's backtest).
    status2 = cs.update_equity(206_000)
    print(f"Phase 2 dip:   equity=${cs.equity:,.0f}  max_dd_floor=${cs.dd_floor_dollars:,.0f}  "
          f"busted={status2['busted']}  newly_breached={status2['newly_breached']}  "
          f"distance_to_floor=${status2['distance_to_max_dd_floor']:,.0f}")
    return status2


def test_static_floor_never_ratchets_and_survives_phase2_dip():
    """STATIC (this account's confirmed real terms): the floor must sit at
    exactly $180,000 the whole time, unmoved by the $232,000 peak, and a
    normal $206,000 Phase-2 dip must NOT bust the account regardless of
    which DrawdownFloorMode is set (it should be irrelevant under STATIC)."""
    for mode in (DrawdownFloorMode.RESET, DrawdownFloorMode.CARRY_OVER):
        cfg = HyroTraderComplianceConfig(drawdown_floor_mode=mode, drawdown_type=DrawdownType.STATIC)
        cs = ComplianceState.start_phase1(cfg)
        cs.start_new_day(dt.date(2026, 1, 1))
        assert cs.dd_floor_dollars == 180_000.0
        for eq in (205_000, 215_000, 232_000, 221_000):
            status = cs.update_equity(eq)
        assert cs.dd_floor_dollars == 180_000.0, \
            f"STATIC floor moved to {cs.dd_floor_dollars} after a peak -- should never ratchet"
        assert not status["busted"]

        cs.advance_to_phase2()
        assert cs.dd_floor_dollars == 180_000.0, \
            f"STATIC floor should reopen at $180,000 in Phase 2 regardless of floor_mode={mode.value}"
        cs.start_new_day(dt.date(2026, 2, 1))
        status2 = cs.update_equity(206_000)
        assert not status2["busted"], \
            f"STATIC floor_mode={mode.value}: a $206,000 Phase-2 dip should NOT bust (floor is $180,000)"
    print("PASS: test_static_floor_never_ratchets_and_survives_phase2_dip")


def test_trailing_reset_vs_carry_over_still_diverge():
    """Regression coverage for the account-downgrade case: TRAILING must
    still reproduce the original divergent RESET vs CARRY_OVER behavior."""
    reset_status = run_scenario(DrawdownFloorMode.RESET, DrawdownType.TRAILING)
    carry_status = run_scenario(DrawdownFloorMode.CARRY_OVER, DrawdownType.TRAILING)

    print("\n" + "=" * 78)
    print("SUMMARY (TRAILING only): same Phase 1 path (peak $232,000), same Phase 2 dip to $206,000")
    print("=" * 78)
    print(f"RESET mode:      floor=$180,000 (90% of fresh $200k)   distance=${reset_status['distance_to_max_dd_floor']:,.0f}   busted={reset_status['busted']}")
    print(f"CARRY_OVER mode: floor=$212,000 (peak $232,000 - $20,000 cushion carried over)   distance=${carry_status['distance_to_max_dd_floor']:,.0f}   busted={carry_status['busted']}")

    assert reset_status["busted"] is False
    assert carry_status["busted"] is True
    print("PASS: test_trailing_reset_vs_carry_over_still_diverge")


def test_default_config_is_static():
    """The account's confirmed real terms (2026-08-10) are static drawdown --
    the dataclass default must reflect that, not silently fall back to the
    stricter TRAILING behavior."""
    cfg = HyroTraderComplianceConfig()
    assert cfg.drawdown_type == DrawdownType.STATIC
    print("PASS: test_default_config_is_static")


if __name__ == "__main__":
    test_default_config_is_static()
    test_static_floor_never_ratchets_and_survives_phase2_dip()
    test_trailing_reset_vs_carry_over_still_diverge()
    print("\nALL COMPLIANCE TESTS PASSED (STATIC default verified, TRAILING regression preserved)")
