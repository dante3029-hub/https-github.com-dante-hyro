# Long-only breakout — the bull-market complement

## The problem it solves

The book is dollar-neutral, so it structurally cannot capture a bull run. Over
the full 1,315-day window:

| | bull half (+157% market) | bear half (-28% market) |
|---|---|---|
| neutral book | **0.25** | 2.54 |

It is not that the book loses money in a rally -- it makes much less. In a
strong bull everything rises together, the shorts bleed, and cross-sectional
ranking has less to work with.

## The sleeve

**50-day high breakout, long only, fixed 3xATR stop, 20-bar hold, daily bars.**

| config | Sharpe | GAP | ann% | bull | bear | n |
|---|---|---|---|---|---|---|
| **50d high, fixed stop** | **1.17** | 0.48 | 88.4% | **2.02** | -0.35 | 311 |
| 20d high, fixed stop | 1.14 | **-0.00** | 121.2% | 1.87 | -0.19 | 526 |
| 50d high, TRAILING stop | -0.99 | 0.68 | -25.8% | -0.88 | -1.09 | 311 |
| +30% in 3d, trail | -0.79 | -0.66 | -17.6% | -0.30 | -1.15 | 121 |
| 50d high trail, 60 coins | -0.74 | 0.28 | -31.3% | -0.71 | -0.76 | 564 |

### Three findings

**Trailing stops destroy it.** Every trailing variant is strongly negative --
50d high goes +1.17 to -0.99. You get stopped out of the pullbacks that precede
the real move. Fourth exit result today pointing the same way: do not cap or
chase winners.

**20-day is not a real signal.** Its control matched it EXACTLY (gap -0.00). A
50-day high is a structural event; a 20-day high is routine.

**Percentage triggers are too late.** "+30% in 3 days" scores -0.79. By the
time it has popped 30%, the move is done. The breakout BEFORE the pop pays.

## Combined with the neutral book

Correlation between them: **-0.076**. Genuinely offsetting.

| book | Sharpe | bull | bear |
|---|---|---|---|
| neutral alone | 1.49 | 0.25 | 2.54 |
| breakout alone | 1.17 | 2.02 | -0.35 |
| **60/40 neutral/breakout** | **1.96** | **1.62** | 2.32 |
| 50/50 | 1.95 | 1.86 | 2.12 |
| breakout only when 60d trend up | 1.97 | 1.77 | 2.25 |

**60/40 lifts the bull half from 0.25 to 1.62** while the bear half only falls
from 2.54 to 2.32. Overall 1.49 -> 1.96.

**No regime timing needed** -- switching the breakout on only when the trend is
up gives 1.97 vs 1.96. Just hold both continuously.

## Caveats

- 311 trades over 3.6 years across 24 coins. Thin.
- Gap is 0.48 -- positive but not large.
- The wider 60-coin universe made it WORSE (-0.74), same as every other sleeve.
- ann% and maxDD in the combined table are scaling artifacts of the nz()
  normalisation, not real returns. Only Sharpe and the regime splits are real.
