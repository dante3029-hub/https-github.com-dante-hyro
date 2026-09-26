# Daily kill switch — $5k/day becomes safe

## The problem

From $214k the static floor is barely a risk (0.4-2%). **The $10,000 daily
limit is what binds.** At $5,000/day there is an 8.3% chance of a -$10k day
before reaching target; at $6,000/day it is 14.2%. That is an instant fail,
not a drawdown you recover from.

## The fix

**Flatten all positions when the day is down $5,000.** You cannot reach -$10k
if you are flat at -$5k.

| vol/day | kill at | daily-fail | floor | pass | median days |
|---|---|---|---|---|---|
| $4,000 | none | 2.67% | 0.52% | 94.5% | 5 |
| $4,000 | -$5k | **0.00%** | 0.29% | 97.9% | 5 |
| $5,000 | none | **8.27%** | 0.65% | 90.3% | 3 |
| **$5,000** | **-$5k** | **0.00%** | 0.26% | **98.9%** | 4 |
| $6,000 | none | 14.23% | 0.51% | 85.1% | 3 |
| $6,000 | -$5k | 0.00% | 0.30% | 99.3% | 3 |

**$5,000/day with a -$5,000 daily kill switch: 98.9% pass, zero daily-limit
risk, 0.26% floor risk, median 4 days to target.**

## Why this is NOT the kill switch that failed earlier

The earlier test PAUSED TRADING FOR DAYS after a loss. That doubled the failure
rate -- you sit flat while the strategy recovers. This one closes positions for
the remainder of a SINGLE DAY and resumes at the next open. Completely
different mechanism.

## Implementation requirements

1. measured **from the day's OPEN**, matching how HyroTrader calculates the
   swing limit -- not from an intraday high-water mark
2. fires on **intraday equity in real time**, not an end-of-day job
3. resumes normally at the next daily boundary
4. logs every trigger -- if it fires more than a few times a month, size is
   too high regardless of what the simulation says

## Sizing decision

**$5,000/day WITH the -$5,000 kill switch.** Without it, $3,000 is the ceiling.
The kill switch is what makes the larger size defensible, so it is not
optional -- it is the reason the size works.
