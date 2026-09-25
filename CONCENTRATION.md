# ZEC concentration — diagnosed, not a sizing problem

## The concern

ZEC is 17.3% of the book's gross profit. Removing it takes the book 3.81 ->
3.23. Removing the top five takes it to 2.49.

## What it is NOT

**Not oversized positions.** ZEC's average exposure is 5.5% against a 4.2%
equal-weight average. Barely above.

**Not fixable by capping.** Tightening the portfolio per-coin cap barely moves
it, because the cap almost never binds:

| coin cap | Sharpe | ZEC% | minus-ZEC |
|---|---|---|---|
| 0.15 (current) | 3.81 | 17.3% | 3.23 |
| 0.04 | 3.79 | 15.9% | 3.27 |
| 0.02 | 3.63 | 13.6% | 3.16 |

Going from 0.15 to 0.02 costs 0.18 Sharpe to move ZEC from 17.3% to 13.6%. You
would be capping a normal-sized position to stop it being right.

## What it IS — per-coin profit vs risk carried

| coin | profit% | exposure% | risk% | profit/risk |
|---|---|---|---|---|
| **ZEC** | 17.3% | 5.5% | **7.9%** | **2.19** |
| 1000PEPE | 9.7% | 4.1% | 5.7% | 1.68 |
| NEAR | 8.7% | 4.2% | 5.4% | 1.63 |
| SUI | 7.3% | 4.1% | 4.8% | 1.52 |
| XLM | 5.2% | 3.7% | 3.9% | 1.35 |
| SOL | 5.0% | 4.6% | 4.3% | 1.16 |
| BCH | 5.4% | 5.2% | 4.8% | 1.12 |
| 1000SHIB | 4.5% | 4.6% | 4.2% | 1.05 |
| ADA | 0.8% | 4.0% | 3.7% | 0.22 |
| XRP | -0.1% | 3.4% | 2.8% | -0.05 |
| LTC | -0.6% | 4.6% | 3.0% | -0.19 |

ZEC carries **7.9% of the risk** because it is genuinely more volatile (1.21%
hourly vs ~0.9% typical). It earned **2.19x** its risk share.

That is real outperformance -- but it sits at the TOP OF A DISTRIBUTION, not
alone: 1000PEPE 1.68, NEAR 1.63, SUI 1.52. And the bottom of the distribution
is negative (LTC -0.19). Dispersion across 24 coins is expected.

## The realistic downside

Zeroing ZEC (3.23) is the WORST case, not the expected one. If ZEC merely
performs **proportionally** next time -- 1.0 instead of 2.19 -- you lose about
half its excess, landing near **3.5**.

## The actual fix: widen the EVENT sleeves

Tested: cross-sectional on 63 coins, events on 25 (all that have OI).

| configuration | Sharpe | maxDD | ZEC% | top5% |
|---|---|---|---|---|
| both on 24 | 3.81 | 4.4% | 17.3% | 48.4% |
| xs on 63, events on 25 | 3.34 | 6.4% | 11.9% | 34.4% |

Concentration falls but it costs 0.47 and raises drawdown. Poor trade on those
odds -- you give up 0.47 in the good case to gain 0.16 in the bad one.

**The event sleeves are the constraint.** They are 4 of 8, they hold the
strongest edge (sr 1.95, fvg 2.96, pattern 1.87, srflip 1.69), and they are
stuck at 25 coins because only 25 have OI data.

```bash
cd ~ && python3 -u fetch_oi.py --run
```

Widening those is the one action that spreads risk without trimming the winner.
Untested until the data exists.
