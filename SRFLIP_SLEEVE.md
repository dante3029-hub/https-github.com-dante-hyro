# S/R flip — found from a chart-reading intuition, not a sweep

## What it is

After price breaks resistance, that level FLIPS to support. Buy when price
returns to it and holds. `res_is_sup` in the ChartPrime source -- present all
along, never tested.

Different trigger from `sr_hourly`, which buys the BREAK. This buys the RETEST,
so it enters on weakness rather than strength.

**6h bars, 3xATR stop, 15-bar hold, 20-bar window for the retest. 450 trades.**

## Standalone

| variant | Sharpe | bull | bear | maxDD | n |
|---|---|---|---|---|---|
| **flip retest 6h, hold 15** | **1.71** | **2.38** | 0.99 | **30.6%** | 450 |
| flip retest 6h, hold 30 | 1.55 | 1.97 | 1.16 | 35.7% | 449 |
| flip retest 12h, hold 30 | 1.23 | 1.46 | 0.97 | 33.3% | 219 |

**bull 2.38 is the highest of any bull candidate tested** -- above breakout-100d
(2.06) and xs-momentum (2.32, but at a 97% drawdown).

## In the book

| book | Sharpe | bull | bear | maxDD |
|---|---|---|---|---|
| 7 sleeves | 3.97 | 3.44 | 4.47 | 6.3% |
| **8 (+ srflip)** | 3.93 | **3.59** | 4.26 | **4.6%** |

**Costs 0.04 Sharpe, gains 0.15 in the bull, cuts max drawdown 27%.**

On a prop account with a STATIC floor the drawdown reduction is worth more than
the Sharpe. The failure mode is touching $180,000, not earning less.

### Why it helps where breakout did not

Both correlate 0.56 with `sr`. The difference is TIMING: breakout used the same
levels AND the same moment (the break). The flip holds at different times -- it
is waiting for the retest while sr is already in.

| srflip vs | corr |
|---|---|
| sr | 0.56 |
| pattern | 0.26 |
| delta | -0.16 |
| everything else | 0.05 - 0.13 |

## What did NOT work — 17 bull candidates tested

| mechanism | best | why not |
|---|---|---|
| breakout (20/50/100d) | 1.71 | 0.56 corr with sr, same timing; book 3.97 -> 3.77 |
| xs momentum (30/60/90d) | 1.32 | 97% maxDD |
| trend filter on basket | 0.52 | pure beta, weak |
| **dip buying** (3 depths x 3 stops) | 0.85 | fails at EVERY stop width, 77-96% maxDD |
| OI expanding + 20d high | 1.21 | below every book sleeve |
| laggard rotation | -0.00 | worthless |

Dip-buying was half of the original intuition and it does not survive. The flip
half does.
