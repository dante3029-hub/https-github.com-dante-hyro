"""
Consolidated bot config. Imports LOCKED constants from portfolio_layer
rather than duplicating them (single source of truth) and loads secrets
from environment variables ONLY -- never hardcoded.

FIX vs the original bot's config.py: that file hardcoded live API keys as
literal strings in source. That is a real security anti-pattern (keys end
up in git history, backups, logs) and is NOT reproduced here.
"""
import os
import datetime as dt
from dataclasses import dataclass

from portfolio_layer.risk_overlay import L_DEFAULT, ORIG_THROTTLE, KILL_SWITCH_DOLLARS, DAILY_LOSS_LIMIT
from portfolio_layer.compliance import DrawdownFloorMode, DrawdownType, HyroTraderComplianceConfig
from portfolio_layer.portfolio import (REFERENCE_NOTIONAL, SLEEVE_NAMES, DEFAULT_STOP_FRAC,
                                        MAX_LOSS_PER_TRADE, LOW_CAP_EXPOSURE_LIMIT,
                                        AGGREGATE_MAX_LOSS)


# ---------------------------------------------------------------- exchange
BYBIT_API_KEY = os.environ.get("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.environ.get("BYBIT_API_SECRET", "")
BYBIT_USE_DEMO = os.environ.get("BYBIT_USE_DEMO", "true").lower() == "true"
# NOTE (AUDIT_FINDINGS.md #2): the original bot's "TESTNET_URL" actually
# pointed at Bybit's real-money-adjacent DEMO trading domain
# (api-demo.bybit.com), not Bybit's actual testnet. That distinction matters
# -- demo trading uses live market data and a simulated ledger; testnet uses
# a separate sandbox market. bot/exchange_client.py surfaces which one is
# selected explicitly instead of letting a "testnet=True" flag hide it.
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# ---------------------------------------------------------------- account / compliance
HYRO_COMPLIANCE_CONFIG = HyroTraderComplianceConfig(
    initial_balance=200_000.0,   # phase-start balance -- HyroTrader floor is 10% BELOW THIS = $180,000. NOT current equity.
    max_drawdown_pct=0.10,
    daily_loss_limit_dollars=DAILY_LOSS_LIMIT,   # 10_000.0, imported from risk_overlay (single source of truth)
    drawdown_type=DrawdownType.STATIC,   # CONFIRMED 2026-08-10 (user statement): paid Swing/static
                                          # drawdown upgrade -- floor fixed at 90% of phase-start
                                          # balance, never ratchets. See compliance.py docstring.
                                          # Switch back to TRAILING if the account is ever downgraded.
    drawdown_floor_mode=DrawdownFloorMode.RESET,  # only matters if drawdown_type is ever switched to
                                                   # TRAILING -- no effect under STATIC. Still UNRESOLVED
                                                   # per compliance.py docstring for the TRAILING case.
    min_trading_days=5,   # STRATEGY.md #44a: Phase 1 = 10 days min, Phase 2 = 5 days min (confirmed via
                           # HyroTrader FAQ during Phase 2 carry-over research). Set per active phase by the
                           # caller (bot/orchestrator.py) -- this default covers Phase 2.
)

# ---------------------------------------------------------------- risk overlay (imported, not duplicated)
LEVERAGE_L = 2.65    # 2026-08-25: raised from 1.70
THROTTLE_BANDS = ORIG_THROTTLE              # [(0.04,1.0),(0.08,0.5),(1.0,0.30)]
KILL_SWITCH_DOLLARS_ = KILL_SWITCH_DOLLARS  # -$3,000 intraday
DAILY_LOSS_LIMIT_ = DAILY_LOSS_LIMIT        # $10,000

# ---------------------------------------------------------------- portfolio
REFERENCE_NOTIONAL_ = REFERENCE_NOTIONAL    # $200,000
SLEEVE_NAMES_ = SLEEVE_NAMES                # ("main","short","flow","delta","relvol","bos")
AB_BLEND_WINDOW = 30
# MEASURED 2026-08-09 -- this is NOT a dormant safety cap. On real sleeve
# history it BINDS ON 100% OF DAYS for delta, relvol and bos (never for
# main/short/flow, which peak at 0.2674 and cannot reach it). It removes
# 54.76% of gross book size on average, and because it clips only the
# Strategy B sleeves it changes the BLEND, not just the size:
#     cap=1.0    ann vol 10.85%   Sharpe 1.5161   CAGR 30.06%
#     uncapped   ann vol 15.45%   Sharpe 2.3380   CAGR 78.51%
# The backtest (Sharpe 2.475, ann vol 14.62%) is reproduced by the UNCAPPED
# path -- so leaving this at 1.0 means trading a materially different, lower
# Sharpe strategy than the one that was validated, and the 46-day median
# time-to-pass model does not apply.
# 2026-08-10: account owner explicitly instructed removing this cap. Set to a
# large finite value (effectively uncapped, matches the 1e9 value already
# tested against the current 819-day 6-sleeve series). This roughly doubles
# gross exposure on delta/relvol/bos vs the cap=1.0 behavior above -- unlevered
# maxDD on the 819-day history rises from 16.87% (cap=1.0) to 20.62%
# (uncapped), and days-to-pass bust rate (no kill switch) rises from 2.57% to
# 4.71% before the kill switch offsets it back down. This is a risk decision
# made by the account owner, not a code fix -- if you revert it, restore 1.0
# and re-read STOP_SYNC_AND_SIZING_FINDINGS.md first.
MAX_SLEEVE_MULTIPLIER = 1e9

# ---------------------------------------------------------------- rebalance cadence, per STRATEGY.md
MAIN_CADENCE_HOURS = 72
FLOW_CADENCE_HOURS = 168     # weekly
DELTA_CADENCE_HOURS = 168
RELVOL_CADENCE_HOURS = 168
SHORT_CHECK_HOURS = 4        # event-driven, checked every 4h bar
BOS_CHECK_HOURS = 4

# ---------------------------------------------------------------- coin -> Bybit symbol mapping
def to_bybit_symbol(coin: str) -> str:
    """Universe B coins (e.g. '1000PEPE', 'ADA') already match Bybit's
    linear-perp contract naming; universe A coins are bare tickers. Both
    just need 'USDT' appended -- confirmed by inspecting both coin lists
    against Bybit's own contract naming convention. Flagged, not assumed
    silently: if a coin is ever added that does NOT map this simply
    (delisted, renamed, or a non-USDT-margined contract), this function
    will produce a symbol Bybit rejects -- exchange_client surfaces that
    as an explicit order-placement error, it does not silently skip it."""
    return f"{coin}USDT"


# ---------------------------------------------------------------- runtime paths
WORKSPACE = os.environ.get("HYRO_WORKSPACE", "/home/user/workspace")
STATE_DIR = os.path.join(WORKSPACE, "bot_runtime")
STATE_FILE = os.path.join(STATE_DIR, "bot_state.json")
LOG_FILE = os.path.join(STATE_DIR, "bot.log")


# ---------------------------------------------------------------- live execution
# Protective per-position stop attached on every newly opened or increased
# position. This carries RULE 4 of the strategy spec ("PER-LEG STOP-LOSS ...
# alt shorts have unbounded loss") into live trading.
#
# NOTE ON THE NUMBER: the backtest's STOP_FRAC = 0.60 is a stop on the
# INDIVIDUAL LEG's own move (a coin moving X% against the entry). It is NOT a
# X% account stop.
#
# 2026-08-10: DECOUPLED from DEFAULT_STOP_FRAC on account-owner instruction.
# LIVE_STOP_LOSS_FRAC now independently controls the real protective stop
# order placed at execution (tightened to 0.24, matching reference_impl's
# backtested STOP_FRAC=0.24, so live execution now matches what the backtest
# assumes -- this was NOT true before this change, see reference_impl.py
# comment for the intrabar-slippage caveat on this value). DEFAULT_STOP_FRAC
# in portfolio_layer/portfolio.py stays at 0.60 UNCHANGED -- it is used only
# as the denominator for the 3%-rule max-loss-per-trade notional cap
# (max_loss_per_trade / stop_frac = $6,000 / 0.60 = $10,000/leg) and is
# intentionally NOT tied to this constant, so tightening the live stop here
# does not loosen that compliance cap.
# 2026-08-10 CHANGED 0.24 -> 0.40 after forensic review.
#
# WHY. The 0.24 setting was chosen because tightening the stop appeared to raise
# Sharpe. Two measurement errors inflated that signal:
#   1. reference_impl.run_sleeve ERASED the loss on the trigger day (zeroed the
#      weight before applying that day's return) instead of booking it. The
#      tighter the stop, the more losses were erased. Measured inflation at 0.24:
#      +0.92 Sharpe. At 0.60: +0.28.
#   2. Even with that fixed, the model books the loss AT the stop level, i.e. it
#      assumes a perfect fill. The project's own intrabar test measured a median
#      3.75% extra adverse slippage per stop-out, and 42% of stops firing a full
#      day earlier than the daily-close model assumes.
#
# With the loss booked AND intrabar slippage priced in, the advantage disappears:
#      stop   fires/yr   modelled   with 3.75% slippage
#      0.60      1.7       1.75            1.73
#      0.40      3.9       1.88            1.82
#      0.24     18.7       2.16            1.90
#      0.20     30.9       2.30            1.88   <- goes BACKWARDS
# and time-to-pass is IDENTICAL at 0.40 and 0.24 (E[29] vs E[28] @ $4k/day).
#
# 0.40 fires 3.9x/yr vs 18.7x/yr -- roughly 5x less dependence on stop fills
# behaving as modelled -- and sits inside the documented 0.40-0.80 plateau.
# The sweep is monotonic all the way to 4.93 Sharpe at a 5% stop, which is proof
# the tightening "gain" is a modelling artifact and not edge.
LIVE_STOP_LOSS_FRAC = 0.24  # decoupled from DEFAULT_STOP_FRAC -- see comment above

# Refuse to place any single order larger than this fraction of account equity.
# A sizing bug that produces a 10x target is otherwise indistinguishable from a
# legitimate order at the exchange boundary.
MAX_SINGLE_ORDER_EQUITY_FRAC = 0.50

# ---------------------------------------------------------------- HyroTrader compliance caps (2026-08-10)
# Both rules below were supplied verbatim by the account owner and cross-checked
# against this codebase: NEITHER had any enforcement anywhere before this. See
# STRATEGY_SPECIFICATION.md cross-check writeup for the full audit trail.

# "the realized loss on any individual trade must not exceed 3% of the initial
# account balance" -- 3% of $200,000 = $6,000. HyroTrader states this rule is
# NOT monitored by their automated system (manual review only) -- enforced here
# by capping NOTIONAL per leg (max_loss_per_trade / stop_frac), never by
# tightening LIVE_STOP_LOSS_FRAC itself. See portfolio_layer.portfolio's
# DEFAULT_STOP_FRAC docstring for why tightening the stop instead was rejected
# (monotonic-improvement-as-stop-tightens is the overfit signature flagged
# earlier this session).
MAX_LOSS_PER_TRADE_ = MAX_LOSS_PER_TRADE  # $6,000.0

# "you must not allocate more than 5% of your initial account balance
# (including leverage) across all low-cap assets at any time" -- 5% of
# $200,000 = $10,000, summed across ALL sleeves.
LOW_CAP_EXPOSURE_LIMIT_ = LOW_CAP_EXPOSURE_LIMIT  # $10,000.0

# 2026-08-10: aggregate correlated-crash cap (all open legs stopping out
# together). NOT part of HyroTrader's stated rules -- this codebase's own
# risk control, added because the per-trade cap above is not pooled across
# the ~34 typical concurrent legs. See portfolio_layer.portfolio's
# AGGREGATE_MAX_LOSS docstring for the exact numbers and rejected alternative.
AGGREGATE_MAX_LOSS_ = AGGREGATE_MAX_LOSS  # $30,000.0 (15% of $200k)

# HyroTrader's low-cap definition: market cap < $100M, OR 24h volume in the
# $500K-$5M band, OR Innovation Zone listing. reference_impl.select_universe()'s
# RULE 1 (MIN_LIQ = $5,000,000 MEDIAN daily volume) is a pure liquidity floor --
# it does NOT check market cap and does NOT check Innovation Zone status, so it
# does not screen HyroTrader's low-cap definition out of the tradeable universe.
#
# This set is a MANUAL, POINT-IN-TIME SNAPSHOT from live web research on
# 2026-08-10 -- NOT a live market-cap feed:
#   ORDI:  market cap $70.44M per Bybit's own price page as of 2026-08-05 --
#          confirmed < $100M. 24h volume ($10-22M across venues) clears the
#          $5M liquidity floor easily, so select_universe() does not exclude it.
#   BRETT: market cap $62M-$93.5M across CoinStats/MarketCapOf/LiveCoinWatch as
#          of 2026-08-10 -- consistently < $100M across every source checked.
# The remaining ~24 universe coins (AAVE, APT, ARB, AVAX, CRV, ENA, ETHBTC, FIL,
# HYPE, ICP, INJ, JTO, JUP, LINK, NEAR, ONDO, OP, PENDLE, SEI, SUI, TIA, WLD,
# XRP, ZEC) were NOT individually re-verified against current market cap this
# pass -- most are almost certainly > $100M but this was not exhaustively
# checked, and market caps drift over time. This hardcoded set MUST be
# refreshed against a live market-cap/listing-tier feed before real trading --
# it is a stopgap for the missing screen in select_universe(), not a
# substitute for building that screen properly.
LOW_CAP_COINS = {"ORDI", "BRETT"}
