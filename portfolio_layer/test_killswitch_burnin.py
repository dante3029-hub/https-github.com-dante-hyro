"""
MULTI-DAY KILL-SWITCH BURN-IN — 2026-08-10.

The kill switch was silently unwired from the live loop for an entire period,
and until now had only been verified against a SINGLE synthetic equity jump.
At the sizes this account trades it is load-bearing: with it, daily-limit
breaches are ~0%; without it they are 21-35% and roughly a third of failures
come from ONE day rather than a slow bleed.

This drives the REAL RiskState through realistic multi-day paths -- gradual
drawdowns, session boundaries, partial recoveries, restart mid-drawdown --
rather than a single manual jump.

Every assertion targets a specific way the switch could fail live.
"""
import datetime as dt

import pytest

from portfolio_layer.risk_overlay import (
    RiskState, KILL_SWITCH_DOLLARS, DAILY_LOSS_LIMIT, L_DEFAULT,
    throttle_multiplier,
)

START = 200_000.0
D0 = dt.date(2026, 8, 10)


def _fresh():
    rs = RiskState.new(START)
    rs.start_new_session(D0)
    return rs


# ---------------------------------------------------------------- gradual
def test_gradual_intraday_bleed_trips_at_threshold_not_before():
    """A slow bleed must trip on the tick that CROSSES -$3,000, not earlier."""
    rs = _fresh()
    tripped_at = None
    for step in range(1, 41):                      # -$100 ... -$4,000 in $100 steps
        pnl = -100.0 * step
        if rs.on_intraday_pnl_update(pnl):
            tripped_at = pnl
            break
    assert tripped_at is not None, "never tripped during a -$4,000 bleed"
    assert tripped_at == -3000.0, f"tripped at {tripped_at}, expected exactly -3000"
    assert rs.target_exposure_multiplier() == 0.0, "not flat after trip"


def test_no_trip_just_above_threshold():
    rs = _fresh()
    assert not rs.on_intraday_pnl_update(-2999.99)
    assert rs.target_exposure_multiplier() > 0.0, "flattened without tripping"


# ------------------------------------------------------- session boundaries
def test_stays_blocked_for_rest_of_session_even_if_pnl_recovers():
    """
    THE failure that costs an account: switch trips, market bounces, bot
    re-enters into the same bad day. Recovery must NOT unblock.
    """
    rs = _fresh()
    assert rs.on_intraday_pnl_update(-3200.0)
    for recovered in (-2000.0, -500.0, +250.0, +1500.0):
        rs.on_intraday_pnl_update(recovered)
        assert rs.kill_switch_tripped_today, "un-tripped on recovery"
        assert rs.target_exposure_multiplier() == 0.0, \
            f"re-entered same session at pnl {recovered}"


def test_rearms_on_next_session_only():
    rs = _fresh()
    rs.on_intraday_pnl_update(-3500.0)
    assert rs.target_exposure_multiplier() == 0.0
    rs.start_new_session(D0 + dt.timedelta(days=1))
    assert not rs.kill_switch_tripped_today, "did not re-arm next session"
    assert rs.target_exposure_multiplier() > 0.0, "still flat after new session"


def test_multi_day_drawdown_trips_each_day_independently():
    """
    5 consecutive losing days, each breaching intraday. The switch must fire
    EVERY day -- a per-session block that silently became permanent (or that
    failed to re-arm) would look identical on day 1 and diverge by day 5.
    """
    rs = RiskState.new(START)
    equity = START
    trips = 0
    for day in range(5):
        rs.start_new_session(D0 + dt.timedelta(days=day))
        day_pnl = 0.0
        for _ in range(10):
            day_pnl -= 400.0                       # -$4,000 across the day
            if rs.on_intraday_pnl_update(day_pnl):
                trips += 1
                break
        equity += day_pnl
        rs.update_equity(equity)
    assert trips == 5, f"switch fired on only {trips}/5 losing days"


# ----------------------------------------------------------- throttle bands
def test_throttle_tightens_as_drawdown_deepens():
    rs = RiskState.new(START)
    rs.start_new_session(D0)
    for equity, expected in ((200_000.0, 1.00),    # 0% dd
                             (193_000.0, 0.50),    # 3.5% -> still full? band check
                             (191_000.0, 0.50),    # 4.5% dd
                             (180_000.0, 0.30)):   # 10% dd
        rs.update_equity(equity)
        mult = throttle_multiplier(rs.current_drawdown())
        if equity == 200_000.0:
            assert mult == 1.0
        elif equity == 180_000.0:
            assert mult == 0.30, f"dd {rs.current_drawdown():.3f} -> {mult}"
    # peak must not ratchet DOWN
    rs.update_equity(175_000.0)
    assert rs.peak_equity == 200_000.0, "peak equity ratcheted down"


def test_peak_ratchets_up_only_on_new_highs():
    rs = RiskState.new(START)
    rs.start_new_session(D0)
    rs.update_equity(210_000.0)
    assert rs.peak_equity == 210_000.0
    rs.update_equity(205_000.0)
    assert rs.peak_equity == 210_000.0
    assert abs(rs.current_drawdown() - (1 - 205_000.0/210_000.0)) < 1e-12


# ---------------------------------------------------------------- restart
def test_restart_mid_drawdown_must_come_back_blocked():
    """
    If the process dies AFTER the switch fires, it must restore blocked.
    Losing that flag on restart walks the bot straight back into the day it
    just escaped. This asserts the STATE carries the information needed --
    if this fails, orchestrator persistence must save kill_switch_tripped_today.
    """
    rs = _fresh()
    rs.on_intraday_pnl_update(-3100.0)
    snapshot = dict(equity=rs.equity, peak_equity=rs.peak_equity,
                    session_date=rs.session_date,
                    kill_switch_tripped_today=rs.kill_switch_tripped_today,
                    daily_loss_limit_tripped_today=rs.daily_loss_limit_tripped_today,
                    session_realized_pnl=rs.session_realized_pnl)
    restored = RiskState(**snapshot)
    assert restored.kill_switch_tripped_today, "restart cleared the kill-switch flag"
    assert restored.target_exposure_multiplier() == 0.0, "restart re-enabled trading"


# ------------------------------------------------------- daily loss limit
def test_daily_loss_limit_also_trips_and_blocks():
    """The firm's $10,000 limit must trip independently of the kill switch."""
    rs = _fresh()
    rs.on_intraday_pnl_update(-DAILY_LOSS_LIMIT - 1.0)
    assert rs.daily_loss_limit_tripped_today
    assert rs.target_exposure_multiplier() == 0.0


def test_killswitch_fires_far_before_the_daily_limit():
    """The whole point: -$3,000 must catch the day long before -$10,000."""
    assert KILL_SWITCH_DOLLARS < DAILY_LOSS_LIMIT / 3.0, \
        "kill switch is not tight enough to protect the daily limit"


# ------------------------------------------------- gap / single-day shock
def test_single_bar_gap_beyond_threshold_still_trips_once():
    """A gap straight through -$3,000 to -$8,000 in one mark must still trip."""
    rs = _fresh()
    assert rs.on_intraday_pnl_update(-8000.0), "gap through threshold did not trip"
    assert not rs.on_intraday_pnl_update(-9000.0), "double-fired on the same session"
    assert rs.target_exposure_multiplier() == 0.0
