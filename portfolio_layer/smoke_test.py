"""
End-to-end smoke test of the Phase 3 pipeline: risk_overlay throttle bands,
kill switch state transitions, and portfolio.size_portfolio() with synthetic
sleeve data. This checks the modules RUN and produce internally-consistent
numbers -- it is not a backtest-parity check (see validate_phase3.py for that).
"""
import sys
import datetime as dt
import numpy as np

sys.path.insert(0, '/home/user/workspace')
from portfolio_layer.risk_overlay import RiskState, throttle_multiplier, L_DEFAULT, KILL_SWITCH_DOLLARS
from portfolio_layer.portfolio import size_portfolio, REFERENCE_NOTIONAL

print("=== throttle_multiplier band boundaries ===")
for dd in [0.0, 0.039, 0.04, 0.041, 0.079, 0.08, 0.081, 0.5, 0.99]:
    print(f"  dd={dd:.3f} -> mult={throttle_multiplier(dd)}")
assert throttle_multiplier(0.039) == 1.0
assert throttle_multiplier(0.04) == 0.5   # dd < 0.04 is False at exactly 0.04 -> next band
assert throttle_multiplier(0.079) == 0.5
assert throttle_multiplier(0.08) == 0.30
assert throttle_multiplier(0.99) == 0.30
print("  boundary assertions passed")

print()
print("=== RiskState: peak tracking + kill switch trip/reset across sessions ===")
rs = RiskState.new(200_000.0)
rs.start_new_session(dt.date(2026, 1, 1))
print(f"  day1 start: equity={rs.equity} peak={rs.peak_equity} dd={rs.current_drawdown():.4f} "
      f"mult={rs.target_exposure_multiplier():.4f} (expect {L_DEFAULT}*1.0)")
assert abs(rs.target_exposure_multiplier() - L_DEFAULT * 1.0) < 1e-9

tripped = rs.on_intraday_pnl_update(-1500)
print(f"  after -$1,500 intraday: tripped={tripped} mult={rs.target_exposure_multiplier():.4f}")
assert tripped is False and rs.target_exposure_multiplier() > 0

tripped = rs.on_intraday_pnl_update(-3000)
print(f"  after -$3,000 intraday: tripped={tripped} mult={rs.target_exposure_multiplier():.4f} (expect 0.0)")
assert tripped is True and rs.target_exposure_multiplier() == 0.0

tripped_again = rs.on_intraday_pnl_update(-3200)
print(f"  further loss same session: tripped_again={tripped_again} (expect False, already tripped) "
      f"mult={rs.target_exposure_multiplier():.4f} (expect still 0.0)")
assert tripped_again is False and rs.target_exposure_multiplier() == 0.0

rs.update_equity(200_000 - 3000)
rs.start_new_session(dt.date(2026, 1, 2))
print(f"  next session after reset: mult={rs.target_exposure_multiplier():.4f} "
      f"peak={rs.peak_equity} equity={rs.equity} dd={rs.current_drawdown():.4f}")
assert rs.target_exposure_multiplier() > 0, "kill switch must clear on new session"

# push equity down further via update_equity to move into the 8% throttle band, peak stays at 200k
rs.update_equity(200_000 * (1 - 0.09))
rs.start_new_session(dt.date(2026, 1, 3))
print(f"  after -9% drawdown from peak: dd={rs.current_drawdown():.4f} "
      f"mult={rs.target_exposure_multiplier():.4f} (expect {L_DEFAULT}*0.30)")
assert abs(rs.target_exposure_multiplier() - L_DEFAULT * 0.30) < 1e-6

print()
print("=== size_portfolio() with synthetic 3-coin sleeves ===")
rng = np.random.default_rng(0)
hist_len = 40
sleeve_hist = {name: rng.normal(0, 0.01, hist_len) for name in
               ("main", "short", "flow", "delta", "relvol", "bos")}
raw_weights = {
    "main": {"BTC": 0.5, "ETH": -0.5},
    "short": {"SOL": -1.0},
    "flow": {"BTC": 0.3, "SOL": -0.3},
    "delta": {"ETH": 0.4, "AVAX": -0.4},
    "relvol": {"BTC": 0.2, "DOGE": -0.2},
    "bos": {"AVAX": -1.0},
}
rs2 = RiskState.new(200_000.0)
rs2.start_new_session(dt.date(2026, 1, 1))
result = size_portfolio(raw_weights, sleeve_hist, rs2, window=30, L=L_DEFAULT)

print(f"  sleeve multipliers: { {k: round(v, 5) for k, v in result.sleeve_multipliers.items()} }")
print(f"  account multiplier: {result.account_multiplier:.5f} (expect {L_DEFAULT}*1.0 = {L_DEFAULT})")
print(f"  gross notional: ${result.gross_notional:,.2f}")
print(f"  flags: {result.flags}")
for sleeve, coins in result.dollar_targets.items():
    print(f"  {sleeve}: {{" + ", ".join(f'{c}: ${d:,.0f}' for c, d in coins.items()) + "}}")

# sanity: gross notional should be well below REFERENCE_NOTIONAL * L given multiple sub-1 multipliers stacking
assert result.gross_notional < REFERENCE_NOTIONAL * L_DEFAULT * 2, "gross notional implausibly large"
assert result.account_multiplier == L_DEFAULT  # no drawdown, no kill switch -> full 1.0 throttle

print()
print("=== size_portfolio() with kill switch tripped ===")
rs3 = RiskState.new(200_000.0)
rs3.start_new_session(dt.date(2026, 1, 1))
rs3.on_intraday_pnl_update(-3000)
result2 = size_portfolio(raw_weights, sleeve_hist, rs3, window=30, L=L_DEFAULT)
print(f"  account multiplier: {result2.account_multiplier} (expect 0.0)")
print(f"  gross notional: ${result2.gross_notional:,.2f} (expect 0.0)")
print(f"  flags: {result2.flags}")
assert result2.account_multiplier == 0.0
assert result2.gross_notional == 0.0

print()
print("ALL SMOKE TESTS PASSED")
