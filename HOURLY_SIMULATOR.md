# Hourly portfolio simulator — validates the blend at 3.46

## Why the daily version was wrong

The first simulator collapsed 6h, 8h and 12h sleeves onto a DAILY grid, so P&L
was measured close-to-close on daily bars rather than at the actual entry and
exit prices. It reported FVG at -0.30 and BOS at -0.47 and I presented that as
a correction to the blend. It was my conversion, not a finding.

Reconciliation caught it -- if both methods are right the TOTAL P&L must match:

| sleeve | trade-level | daily-marked | gap |
|---|---|---|---|
| sr | 466.9% | 286.6% | **-180.3%** |
| breakout | 2101.0% | 890.4% | **-1210.6%** |

Two causes: the wrong price grid for sub-daily sleeves, and a max-concurrent
rescale that silently shrank positions on busy days.

## What the hourly build does differently

1. every sleeve generates trades on **its own timeframe** (6h / 8h / 12h / daily)
2. positions held on a **31,536-step hourly grid** by actual timestamp
3. entry and exit prices from the **sleeve's own bars**, so a stop triggering on
   a 6h low is booked at that bar
4. **no max-concurrent rescale**

Reconciliation after: breakout 2446% hourly vs 2101% trade-level -- a **16% gap
vs the daily version's 58%**. The residual is `searchsorted` mapping a daily bar
to its opening hour, so holding periods can be off by up to a day.

## Results

| sleeve | Sharpe | ann% | maxDD | mean gross |
|---|---|---|---|---|
| fvg | 3.29 | 549.0% | 91.0% | **2.087** |
| sr | 2.02 | 69.4% | 22.5% | 0.243 |
| delta | 1.41 | 25.8% | 32.9% | 0.850 |
| breakout | 1.34 | 113.3% | 50.6% | 0.601 |
| skew | 1.26 | 20.9% | 17.1% | 0.836 |
| relvol | 1.17 | 15.5% | 14.8% | 0.646 |
| **bos8** | **0.12** | 5.7% | 62.5% | 0.466 |

| portfolio | Sharpe | ann% | maxDD |
|---|---|---|---|
| native size, costs per sleeve | 3.39 | 114.2% | 13.5% |
| **native size, NETTED** | **3.46** | 111.7% | 12.9% |
| **equal-risk, NETTED** | **3.39** | 53.0% | **6.5%** |
| equal-risk, coin cap 0.08 | 3.37 | 52.1% | 6.3% |

**3.46 netted vs the blend's 3.68** -- the blend stands.

## What the per-sleeve view revealed that the blend hid

**FVG runs at 2.087 mean gross**, more than double any other sleeve. Its 549%
annual and 91% drawdown are not skill, they are size. Equal-risk scaling cuts
it to 0.2x.

**BOS8 is 0.12, not 1.22.** Marked hourly on its native 8h grid it barely works.
The daily simulator's negative reading pointed at something real here.

**Netting saves 44.7% of gross** -- nearly double the 23.8% seen with only the
4 cross-sectional sleeves. Event sleeves crowd into the same coins;
cross-sectional ones spread across a ranking.

**Equal-risk scaling halves the drawdown for the same Sharpe** (12.9% -> 6.5%).
That is the version to run.

## Open

1. the 16% reconciliation residual (entry/exit hour alignment)
2. why BOS reads 0.12 here vs 1.22 standalone
