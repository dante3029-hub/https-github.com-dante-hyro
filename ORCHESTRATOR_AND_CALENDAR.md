# Orchestrator v2 (health gate wired) + calendar levels tested

## orchestrator_v2.py — the health check as a HARD GATE

Glue only: it does not compute signals (signal_engine does) or size positions
(portfolio_layer does). It decides which sleeves are due, asks for targets,
runs the health check, and **refuses to trade if anything is CRITICAL**.

### Verified

| test | result |
|---|---|
| normal cycle, 9 sleeves due | runs, gate evaluated |
| **stale data** | **HALTED -- "14 CRITICAL: data STALE"** |
| **a sleeve raises an exception** | **captured as CRITICAL, cycle halts** |
| state JSON round-trip | OK (483 bytes) |
| flat-cycle tracking | correct per sleeve |

The exception case is the important one. **A sleeve that throws is NOT
silently treated as flat** -- that is precisely how BOS hid for 300+ cycles.
It is recorded as CRITICAL and the cycle halts.

### Design notes

**Event sleeves are checked EVERY cycle regardless of cadence.** Their cadence
describes their BAR SIZE, not a rebalance schedule -- an open slot can need
closing at any time.

**Sleeves not due carry their previous target forward untouched**, rather than
returning empty and looking flat to the health check.

---

## Calendar levels — work standalone, duplicate the book

Yearly open, previous week high/low, Monday open, Monday low, previous month
high/low, previous day high/low. All computed from COMPLETED periods only --
last week's high is known the moment this week starts and never changes.

| level / mode | Sharpe | bull | bear | maxDD | n |
|---|---|---|---|---|---|
| **prev_mo_high break** | **1.66** | 2.36 | 0.91 | **23.3%** | 1097 |
| monday_low retest | 1.50 | 2.22 | 0.86 | 63.7% | 2979 |
| prev_day_high break | 1.46 | 2.17 | 0.79 | 56.7% | 8633 |
| year_open break | 1.28 | 2.04 | 0.44 | 46.8% | 827 |
| prev_wk_high break | 1.21 | 1.45 | 0.96 | 58.2% | 2557 |
| this_wk_open break | 1.09 | 1.76 | 0.47 | 69.8% | 6365 |
| prev_wk_low break | 0.44 | 0.59 | 0.34 | 52.3% | 2748 |

Best is **previous month high, break up** -- 1.66 with only 23.3% drawdown,
comparable to sr's 21% and far better than the other calendar levels (46-71%).

### But it is sr in another guise

| vs | corr |
|---|---|
| **sr** | **0.67** |
| srflip | 0.48 |
| pattern | 0.43 |
| fvg | 0.23 |
| delta | -0.19 |

**0.67 is the highest correlation of anything tested.** Book 4.13 -> 3.90,
drawdown 3.7% -> 4.8%.

A monthly high IS a resistance level. sr just finds it through volume-confirmed
pivots instead of a calendar.

### That is now SIX candidates rejected for the same reason

| candidate | standalone | corr to sr | book effect |
|---|---|---|---|
| prev-month-high | 1.66 | **0.67** | -0.23 |
| breakout 100d | 1.71 | 0.56 | -0.04 |
| EMA retest | 1.46 | 0.55 | -0.21 |
| sr on BTC pairs | 1.90 | 0.45 | -0.02 |
| spread momentum | 0.96 | 0.09 | -0.02 (too weak) |
| short-term reversal | -1.24 | -- | negative |

**The "long-only price structure" dimension is saturated.** sr, srflip, pattern
and fvg already occupy it. Anything else measuring the same thing -- whatever
chart or calendar it is derived from -- lands at 0.4-0.7 correlation and
contributes nothing.

The only thing that HAS added since is `oirank`, and it is the only sleeve
reading something other than price or volume.
