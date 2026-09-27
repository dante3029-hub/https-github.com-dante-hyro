# Config for the nine-sleeve book

`bot/config_v2.py` is written as a SEPARATE module so the live bot keeps
running on the old config until you switch deliberately. A half-applied edit
changing behaviour mid-cycle is exactly the failure mode to avoid.

## What changes

| | |
|---|---|
| REMOVED | main, short, flow, bos |
| KEPT | delta (cadence changed), relvol (unchanged) |
| ADDED | skew, cascade, oirank, sr, srflip, pattern, fvg |

**Only delta and relvol survive from the live book.** This is not a tidy-up --
it is a different strategy, and it should be switched to deliberately.

## Cadence

| sleeve | cadence | universe |
|---|---|---|
| delta | **336h (14d)** -- was 168 | WIDE (60) |
| relvol | 168h (7d) | CORE24 |
| skew | 1080h (45d) | CORE24 |
| oirank | 72h (3d) | CORE24 |
| cascade | 24h | CORE24 |
| sr / srflip / pattern | 6h | CORE24 |
| fvg | 12h | CORE24 |

### Why delta moves from 168h to 336h

The IC on matched-venue data strengthens all the way out to 30 days:

| horizon | IC | t-stat |
|---|---|---|
| 1d | 0.0050 | **0.75** (not significant) |
| 7d | 0.0359 | 5.42 |
| 14d | 0.0650 | 9.98 |
| 21d | 0.0793 | 12.05 |
| 30d | 0.0835 | 12.83 |

hold=7 came from a grid search -- exactly what the manuals warn against.
Realised Sharpe follows the IC: hold 14 gives **1.28 at 26x turnover** vs
hold 5's 1.18 at 73x. Blend 3.24 -> 3.44.

The turnover drop matters beyond the Sharpe: at 26x, the never-measured maker
fill rate stops being load-bearing.

## Universes are PINNED, never globbed

`book.py` used to build its universe by globbing `taker_data/`. When the
60-coin fetch landed it **silently switched from 24 coins to 60** and skew
went from +0.88 to -0.16 with no code change.

Sleeves do NOT share a universe. delta benefits from the wide set (1.11 on 24,
1.45 on 60); relvol degrades (1.58 -> 0.93) and skew breaks entirely
(0.88 -> -0.16). Pairing delta-60 with relvol-24 was worth 1.78 -> 2.07.

## TWO RISK ITEMS NEEDING A DECISION, NOT A DEFAULT

### 1. The intraday kill switch is probably too tight

Current `KILL_SWITCH_DOLLARS = -3_000`.

At $5,000/day volatility that is **0.6 sigma**. P(a daily loss that large) is
roughly 27% -- it would fire about **once every four days** and flatten the
book each time. That is not risk control, it is a different strategy.

The modelled figure is **-$5,000** (1.0 sigma, fires on ~16% of days), and it
takes daily-limit failure to ZERO because you cannot reach -$10,000 while flat
at -$5,000:

| vol/day | kill | daily-fail | floor | pass | median days |
|---|---|---|---|---|---|
| $5,000 | none | 8.27% | 0.65% | 90.3% | 3 |
| $5,000 | **-$5k** | **0.00%** | 0.26% | **98.9%** | 4 |

Raising a kill switch is a risk decision. **Flagged, not changed.**

### 2. MAX_SLEEVE_MULTIPLIER = 1e9

Set for the SIX-sleeve book on an explicit instruction, documented at length
in config.py. There are now NINE sleeves, four of them event-driven and flat
much of the time. **The interaction is different and has not been measured.**

## Sizing from $214k

Static floor $180k -> $34k buffer. Target $220k = +$6k away.

| vol/day | floor risk (SR 4.13) | floor risk (SR 2.1) | P(-$10k day) |
|---|---|---|---|
| $2,000 | 0.04% | 1.16% | 0.0% |
| $3,000 | 0.41% | 3.11% | 0.4% |
| $4,000 | 1.17% | 4.88% | 6.3% |
| $5,000 | 2.14% | 6.28% | **23.4%** |

The floor is barely a risk from $214k. **The $10k daily limit is what binds**,
and past $4,000/day it dominates -- which is what the kill switch is for.

`TARGET_DAILY_VOL_DOLLARS = 5000` requires the -$5k kill switch.
`TARGET_DAILY_VOL_CONSERVATIVE = 3000` is safe without it.

## Health check

`HALT_ON_CRITICAL = True`. BOS produced zero weights for 300+ cycles while the
log reported success; `run/` went 408h stale while cycles kept "succeeding";
the taker column had taker_buy > volume on 60% of bars. All three silent.

Run `run_health_check()` at the end of every cycle. If any CRITICAL is open,
log it and place NO orders. **A sleeve that cannot be verified does not trade.**
