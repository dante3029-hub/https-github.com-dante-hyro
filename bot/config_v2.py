"""
config_v2.py — the nine-sleeve book.

Apply these to bot/config.py and portfolio_layer/portfolio.py. Written as a
separate module so the live bot keeps running on the old config until you
switch deliberately, rather than a half-applied edit changing behaviour
mid-cycle.

WHAT CHANGES AND WHY

  REMOVED   main, short, flow, bos
  KEPT      delta (cadence changed), relvol (unchanged)
  ADDED     skew, cascade, oirank, sr, srflip, pattern, fvg

Only delta and relvol survive from the live book. That is not a tidy-up -- it
is a different strategy, validated end to end, and it should be switched to
deliberately rather than merged in piece by piece.
"""

# ════════════════════════════════════════════════════════════════════════
# portfolio_layer/portfolio.py
# ════════════════════════════════════════════════════════════════════════

SLEEVE_NAMES = (
    # cross-sectional, ranked long/short
    "delta",      # taker-flow imbalance, 60 coins
    "relvol",     # relative volume, 24 coins
    "skew",       # 60d realised skewness, 24 coins
    "oirank",     # OI change rank, blended lookbacks
    # event-driven
    "cascade",    # market-wide flush fade, long only
    "sr",         # resistance break, long only
    "srflip",     # resistance-turned-support retest, long only
    "pattern",    # MarkitTick bullish patterns, long only
    "fvg",        # fair value gap, both directions
)

# ════════════════════════════════════════════════════════════════════════
# bot/config.py — rebalance cadence
# ════════════════════════════════════════════════════════════════════════

# DELTA: 168 -> 336. The live bot rebalances weekly (hold 7d). The IC analysis
# on matched-venue data says the signal STRENGTHENS out to 30 days:
#
#     horizon   IC       t-stat
#     1d        0.0050   0.75     <- not significant
#     7d        0.0359   5.42
#     14d       0.0650   9.98
#     21d       0.0793   12.05
#     30d       0.0835   12.83
#
# hold=7 came from a grid search, which is exactly what the manuals warn
# against. Realised Sharpe follows the IC: hold 14 gives 1.28 at 26x turnover
# vs hold 5's 1.18 at 73x. Blend 3.24 -> 3.44.
#
# The turnover drop matters beyond the Sharpe: at 26x the unmeasured maker
# fill rate stops being load-bearing.
DELTA_CADENCE_HOURS = 336         # was 168

RELVOL_CADENCE_HOURS = 168        # unchanged -- research agrees, N=8, hold 7d
SKEW_CADENCE_HOURS = 45 * 24      # 1080. 60d lookback, 45d hold
OIRANK_CADENCE_HOURS = 72         # 3d hold, rank blended over 3/5/7/10/14d

# Event sleeves check every bar on their own timeframe.
CASCADE_CHECK_HOURS = 24
SR_CHECK_HOURS = 6
SRFLIP_CHECK_HOURS = 6
PATTERN_CHECK_HOURS = 6
FVG_CHECK_HOURS = 12

CADENCE_BY_SLEEVE = {
    "delta": DELTA_CADENCE_HOURS,
    "relvol": RELVOL_CADENCE_HOURS,
    "skew": SKEW_CADENCE_HOURS,
    "oirank": OIRANK_CADENCE_HOURS,
    "cascade": CASCADE_CHECK_HOURS,
    "sr": SR_CHECK_HOURS,
    "srflip": SRFLIP_CHECK_HOURS,
    "pattern": PATTERN_CHECK_HOURS,
    "fvg": FVG_CHECK_HOURS,
}

# ════════════════════════════════════════════════════════════════════════
# universes — PINNED, never globbed
# ════════════════════════════════════════════════════════════════════════
#
# book.py used to build its universe by globbing taker_data/. When the 60-coin
# fetch landed it SILENTLY switched from 24 coins to 60 and skew went from
# +0.88 to -0.16 with no code change. Pin them.

CORE24 = (
    "1000PEPE", "1000RATS", "1000SHIB", "AAVE", "ADA", "AVAX", "BCH", "BNB",
    "DOGE", "DOT", "ETH", "FIL", "LDO", "LINK", "LTC", "NEAR", "SOL", "SUI",
    "TRX", "UNI", "WLD", "XLM", "XRP", "ZEC",
)

# delta alone benefits from the wider set (1.11 on 24 coins, 1.45 on 60).
# relvol degrades (1.58 -> 0.93) and skew breaks entirely (0.88 -> -0.16), so
# they stay on the core. There is no rule that sleeves must share a universe,
# and pairing delta-60 with relvol-24 was worth 1.78 -> 2.07.
UNIVERSE_BY_SLEEVE = {
    "delta": "WIDE",
    "relvol": "CORE24",
    "skew": "CORE24",
    "oirank": "CORE24",
    "cascade": "CORE24",
    "sr": "CORE24",
    "srflip": "CORE24",
    "pattern": "CORE24",
    "fvg": "CORE24",
}

# ════════════════════════════════════════════════════════════════════════
# risk — TWO ITEMS NEED A DECISION, NOT A DEFAULT
# ════════════════════════════════════════════════════════════════════════

# 1) THE INTRADAY KILL SWITCH IS PROBABLY TOO TIGHT FOR THE NEW SIZING.
#
# Current: KILL_SWITCH_DOLLARS = -3_000.
#
# At $5,000/day volatility, -$3,000 is 0.6 sigma. P(a daily loss that large)
# is roughly 27%, so it would fire about ONCE EVERY FOUR DAYS and flatten the
# book each time. That is not risk control, it is a different strategy.
#
# The modelled figure is -$5,000, which at $5,000/day vol is 1.0 sigma and
# fires on roughly 16% of days -- and it takes daily-limit failure to ZERO
# because you cannot reach -$10,000 while flat at -$5,000:
#
#     vol/day   kill      daily-fail   floor   pass    median days
#     $5,000    none        8.27%      0.65%   90.3%       3
#     $5,000    -$5k        0.00%      0.26%   98.9%       4
#
# Raising a kill switch is a risk decision. Flagged, not changed.
KILL_SWITCH_DOLLARS_SUGGESTED = -5_000.0

# 2) MAX_SLEEVE_MULTIPLIER = 1e9 (effectively uncapped) was set for the
# SIX-sleeve book on an explicit instruction, documented in config.py. There
# are now NINE sleeves, four of them event-driven and flat much of the time.
# The interaction is different and has not been measured. Re-read
# STOP_SYNC_AND_SIZING_FINDINGS.md before assuming the old reasoning carries.

# ════════════════════════════════════════════════════════════════════════
# sizing, from the block-bootstrap on the 9-sleeve book
# ════════════════════════════════════════════════════════════════════════
#
# Account $214k, static floor $180k -> $34k buffer, target $220k = +$6k away.
#
#     vol/day   floor risk (SR 4.13)   floor risk (SR 2.1)   P(-$10k day)
#     $2,000          0.04%                  1.16%               0.0%
#     $3,000          0.41%                  3.11%               0.4%
#     $4,000          1.17%                  4.88%               6.3%
#     $5,000          2.14%                  6.28%              23.4%
#
# The floor is barely a risk from $214k. The $10k DAILY limit is what binds,
# and past $4,000/day it dominates -- which is exactly what the kill switch
# above is for.
TARGET_DAILY_VOL_DOLLARS = 5_000.0     # requires the -$5k kill switch
TARGET_DAILY_VOL_CONSERVATIVE = 3_000.0  # safe without it

# ════════════════════════════════════════════════════════════════════════
# health check — refuse to trade on CRITICAL
# ════════════════════════════════════════════════════════════════════════
#
# BOS produced zero weights for 300+ live cycles while the log reported
# success. `run/` went 408h stale while cycles kept "succeeding". The taker
# column had taker_buy > total volume on 60% of bars. All three were silent.
#
# run_health_check() at the end of every cycle; if any CRITICAL is open, log
# it and place NO orders. A sleeve that cannot be verified does not trade.
HALT_ON_CRITICAL = True
HEALTH_LOG_ALL_SEVERITIES = True
