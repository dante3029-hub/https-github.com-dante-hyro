# Exit parameters — swept, not defaulted

S/R and FVG were the two strongest sleeves but had never had their exits tested;
both were running a generic 2xATR stop with a 15-bar cap because that is what I
reached for first. Sweeping them is worth +0.15 at blend level.

## Findings

**Take-profits hurt, in every sleeve tested.** Pattern, S/R and FVG all score
worse with a TP at any level. S/R stop3A/h15: 1.58 without, 0.83 with TP 3ATR.
You are capping the winners that pay for the losers.

**Hold 30 is too long.** In FVG every hold-30 row has a weak or negative second
half (0.03, 0.17, 0.36, -0.04).

**Stop direction differs by sleeve type, and it is mechanical:**

| sleeve | best stop | why |
|---|---|---|
| S/R break | **3.0 ATR** | structural break, needs room to develop |
| FVG | **1.0 ATR** | short-horizon imbalance; if it does not resolve fast the premise failed |

S/R shows a clean plateau (3.0A beats 2.0A at every hold). FVG's 1.0A is a
SPIKE -- 2.0A is the worst of four -- so treat 1.44 with more caution than 1.58.

## Settings

| sleeve | timeframe | stop | hold | TP | Sharpe |
|---|---|---|---|---|---|
| sr | 6h break_res +OI | 3.0 ATR | 15 | none | 1.58 |
| fvg | 12h detect +OI | 1.0 ATR | 10 | none | 1.33 |

## Blend

| config | Sharpe | halves |
|---|---|---|
| 4 core + old exits | 2.99 | 2.96 / 3.01 |
| 4 core + NEW exits | **3.14** | 3.23 / 3.04 |
| + bos8 | 3.24 | 3.57 / 2.89 |

bos8 adds 0.10 but its own second half is 0.10 -- it buys headline Sharpe with
stability. The 3.14 with balanced halves is the more honest number.
