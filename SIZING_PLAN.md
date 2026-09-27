# The sizing plan

**$7,000/day. Kill switch at -$5,000 intraday. De-gear to $3,000/day if equity
touches $210,000.**

Validated on EVERY start date across 1,315 days of realised path -- not a
resampled Monte Carlo.

| big size | trigger | passed | failed | median days | worst equity |
|---|---|---|---|---|---|
| **$7,000** | **$210k** | **99.9%** | **0.0%** | **5** | $184,967 |
| $7,000 | none | 99.2% | 0.7% | 5 | **$177,155 BREACH** |
| $6,000 | $210k | 99.9% | 0.0% | 6 | $185,464 |
| $5,000 | $210k | 99.9% | 0.0% | 7 | $186,215 |
| $3,000 | flat | 99.8% | 0.0% | 11 | $189,377 |

## The de-gear is not optional

Without it the same $7,000 size takes worst equity to **$177,155 -- below the
$180,000 floor**. The trigger is doing all the work, not the size.

**$210k is the right level.** Cutting later ($205k, $200k) lets more damage
accumulate first and produces a WORSE worst-case in every row.

## Why 5 days

Only **+$6,000** is needed from $214,000. At $7,000/day that is under one day
of expected return, so the median is 5 days including noise.

## What the end-to-end test found that the earlier sizing work missed

Sized at $5,000/day on the realised path WITHOUT a kill switch:

| | |
|---|---|
| max drawdown | **$41,039** -- exceeds the $34,000 buffer |
| worst day | **-$15,961** |
| days below -$10,000 | **6** -- the hard daily limit, breached |
| days below -$5,000 | 66 of 1,315 (5.0%) |

The kill switch caps those six days. The de-gear caps the drawdown. Neither
alone is enough at $7,000.

## Implementation requirements

1. **Kill switch measured from the DAY'S OPEN**, matching how HyroTrader
   computes the swing limit -- not from an intraday high-water mark.
2. **Fires on intraday equity in real time**, not an end-of-day job.
3. **De-gear checked BEFORE each rebalance**, not once a day. A trigger that
   fires late is not protective.
4. **Log every kill-switch trigger.** At $7,000/day it should fire on roughly
   24% of days. Much more than that and the size is wrong regardless of what
   the simulation says.

```python
def daily_vol_target(equity: float) -> float:
    return (DEGEAR_DAILY_VOL_DOLLARS if equity <= DEGEAR_TRIGGER_EQUITY
            else TARGET_DAILY_VOL_DOLLARS)
```

## Caveat

**In-sample.** This is the same 1,315 days the strategy was built on. The
de-gear structure is sound regardless -- it is a risk rule, not a fitted
parameter -- but the 99.9% pass rate is not a forecast.
