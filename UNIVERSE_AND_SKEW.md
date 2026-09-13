# Universe pinning, sleeve-specific universes, and the skew correction

## The bug that caused all of this

`book.py` built its universe by globbing `taker_data/*.csv`. When the 60-coin
taker fetch was pushed, **book.py silently switched from 24 coins to 60** with
no code change. Skew went from +0.88 to -0.16 and I spent a while calling it
"unstable" when it was fine -- I had changed the universe underneath it.

**Fixed:** `SYMS` is now a pinned CORE24 list, not a glob.

## Sleeve-specific universes — worth +0.29

There is no reason the sleeves must share a universe, and they want opposite
things:

| top N by turnover | delta | relvol | 3-blend |
|---|---|---|---|
| 12 | 0.37 | 1.08 | 1.12 |
| 20 | 0.77 | 1.73 | 1.47 |
| 24 | 1.11 | 1.58 | 1.46 |
| 40 | 1.12 | 1.35 | 1.39 |
| 60 | **1.45** | 0.93 | 1.54 |

**Delta improves with more coins** -- more names, better-populated order-flow
ranking. **Relvol degrades** -- unusual volume is a clean signal among liquid
names and noise on thin ones.

| pairing | Sharpe |
|---|---|
| both on 24 (current) | 1.78 |
| delta 60 + relvol 20 | 1.74 |
| **delta 60 + relvol 24** | **2.07** |

## Skew — I was wrong to call it fragile

I saw skew read 0.05 and recommended dropping it. That reading was on the
60-coin universe with 4 positions. Tested properly on 24 coins it is a clean
plateau:

| lookback | n=3 | n=4 | n=5 | n=6 | n=8 |
|---|---|---|---|---|---|
| 30d | 0.59 | 0.51 | 0.57 | 0.60 | 0.71 |
| 45d | 0.64 | 0.77 | **0.88** | 0.90 | 0.89 |
| **60d** | 0.83 | 0.99 | 1.10 | **1.18** | 1.30 |

Fourteen of fifteen cells have both halves positive. The surface is smooth in
both directions. **60d lookback beats 45d at every position count**, with
control gaps of 1.9-2.95 vs the 45d row's 0.52.

**Change: skew lookback 45d -> 60d, n=5 -> 6.** Gives 1.18 with halves
1.18 / 1.17.

Lesson: test the parameter surface before calling something fragile. A single
low reading in a changed configuration is not evidence of instability.

## Individual coins, buy & hold (context)

| best | Sharpe | ann% | worst | Sharpe |
|---|---|---|---|---|
| ZEC | 1.34 | 151% | S | -1.27 |
| HYPE | 1.20 | 112% | TRUMP | -0.73 |
| TRX | 0.97 | 63% | APT | -0.54 |
| SOL | 0.89 | 73% | POL | -0.50 |

ZEC and HYPE beat the entire neutral book on their own. That is what a bull run
in the right coin does -- and why the breakout sleeve matters.

## Updated sleeve table

| sleeve | config | Sharpe |
|---|---|---|
| sr | 6h break_res +OI, 3ATR, hold15 | 1.58 |
| relvol | **24 coins**, N=8, hold 7 | 1.54 |
| delta | **60 coins**, z7d, hold 14, N=5 | 1.45 |
| fvg | 12h detect +OI, 1ATR, hold10 | 1.33 |
| skew | **24 coins, 60d lookback, n=6** | **1.18** |
| cascade | z <= -2.5, hold 3d | 1.17 |
| breakout | 50d high, long only, 3ATR, hold20 | 0.99 |
