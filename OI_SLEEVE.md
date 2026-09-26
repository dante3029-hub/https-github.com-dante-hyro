# OI cross-sectional rank — the 9th sleeve, book 3.81 -> 4.13

## Why it works where nothing else did

Eight sleeves all read **price or volume**. sr, srflip, pattern, fvg, breakout,
EMA retest, BTC-pair signals -- every one answers "is price continuing up?"
They correlate 0.3-0.6 with each other and adding another changes nothing.

**Open interest is POSITIONING** -- where traders are opening new exposure. It
is the only non-price data in the book.

| vs | corr | vs | corr |
|---|---|---|---|
| delta | -0.05 | sr | 0.15 |
| relvol | 0.31 | pattern | 0.17 |
| skew | 0.01 | fvg | 0.13 |
| cascade | -0.04 | srflip | 0.05 |

## The sleeve

**Long the 5 coins with the biggest OI expansion, short the 5 with the biggest
contraction. Rebalance every 3 days. Rank averaged across 3/5/7/10/14-day
lookbacks.**

| book | Sharpe | bull | bear | maxDD |
|---|---|---|---|---|
| 8 sleeves | 3.81 | 3.43 | 4.18 | 4.4% |
| **9 (+ OI, blended)** | **4.13** | **3.93** | 4.32 | **3.7%** |
| 9 (+ OI, 7d only) | 4.20 | 3.98 | 4.43 | 3.0% |

## Audit

**Control PASSES strongly.** Random coin selection on the same bars with the
same turnover: -1.83 / -1.88 / -1.45 vs the real 1.34. **Gap +2.78 to +3.22.**

**N sweep is a plateau:** 1.09 / 1.19 / 1.34 / 1.33 / 1.05 for N=3..8.

**The LOOKBACK was a spike, and is now blended:**

| lookback | best across holds |
|---|---|
| 3d | 0.92 |
| 5d | 0.97 |
| **7d** | **1.41** |
| 10d | 0.97 |
| 14d | 0.49 |

7d was 45% above either neighbour -- fitted. Averaging the rank across five
lookbacks costs 0.07 in the book and removes the dependence. **The wider blend
(4.13) beats the narrow 5/7/10 one (4.08)**, which is what a real effect does.

**Raw predictive correlation is near zero** -- OI change vs NEXT-day return
averages +0.010 per coin. Same-day is +0.287, which is just the mechanical fact
that OI rises as price rises. The edge is cross-sectional (which coins), not
time-series (when).

## A lookahead bug found and fixed during this work

The first version of the test harness set weights from `sig(t)` and applied
them to `R(t)` -- the SAME bar. OI rises BECAUSE price rose, so it "predicted"
the bar it was measured on. That produced **+4.09**. Lagged properly it is 1.34.

## What else was tested and rejected

| idea | result |
|---|---|
| short-term reversal | -1.24 to -1.81, six configs. IC is real (t=-5 to -6) but costs destroy it |
| BTC-pair ratios as a FILTER | every variant worse than no filter |
| BTC-pair ratios as a SIGNAL | 1.90, but 0.45 corr with sr -> +0.10 bull, -0.02 book |
| BTC ratio predictive power | IC identical to USDT momentum to 3 decimals -- BTC is a common factor, so dividing by it barely changes the cross-sectional ranking |
| EMA retest (50/100/200) | 1.46 best, 0.55 corr with sr, book -0.21 |
| pairs mean-reversion | -0.99 to -1.79 |
| spread momentum | 0.96, orthogonal (0.09) but too weak; book -0.02 |
| hour-of-day | raw effects strong (22:00 = +5.36bp) but -10.37 after taker costs |
