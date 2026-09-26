# Forensic audit of the 7-sleeve book

Ran against the failure modes that have ACTUALLY occurred in this project, not
a generic checklist.

## Result: book 3.97, 46.4% ann, 6.5% maxDD, bull 3.44 / bear 4.47

| sleeve | Sharpe | ann% | maxDD | gross | turn/yr |
|---|---|---|---|---|---|
| fvg | 3.49 | 215.3% | 28.1% | 0.953 | **297.5** |
| sr | 2.02 | 61.5% | 21.4% | 0.229 | 52.6 |
| pattern | 1.81 | 56.7% | 23.8% | 0.271 | 77.4 |
| delta | 1.41 | 25.8% | 32.9% | 0.850 | 29.4 |
| skew | 1.26 | 20.9% | 17.1% | 0.836 | 9.5 |
| relvol | 1.17 | 15.5% | 14.8% | 0.646 | 56.8 |
| cascade | 0.49 | 13.4% | 36.8% | 0.033 | 12.2 |

## 1. LOOKAHEAD — clean

Compared correct `weights.shift(1) * returns` against the cheating no-shift
version. A large gap means the sleeve lives on same-bar information.

| sleeve | correct | no-shift | gap |
|---|---|---|---|
| delta | 1.55 | 1.57 | 0.02 |
| relvol | 1.53 | 1.55 | 0.02 |
| skew | 1.30 | 1.30 | -0.00 |
| cascade | 0.53 | 0.60 | 0.07 |
| sr | 2.15 | 2.34 | 0.19 |
| pattern | 2.09 | 2.43 | 0.34 |
| fvg | 3.54 | 3.96 | 0.42 |

Nothing near the 1.5 threshold. The event sleeves show slightly larger gaps,
expected since the entry bar matters more for them.

## 2. CONCURRENCY — a real bug, now fixed

**`per_position = 1/6` does NOT cap concurrency.** FVG ran at a mean of **12.5
simultaneous positions** -- 2.1x levered -- because nothing stopped more than
six trades being open at once. Up to **6 simultaneous positions landed in the
SAME coin**.

| FVG | before | after cap |
|---|---|---|
| Sharpe | 3.29 | **3.49** |
| annual | 549.0% | 215.3% |
| maxDD | **91.0%** | **28.1%** |
| gross | 2.087 | 0.953 |

Capping RAISED the Sharpe -- the leverage was adding variance faster than
return. `positions_to_hourly` now takes `max_gross` and every event sleeve
passes 1.0.

## 3. DIRECTION CONTROL — passes decisively

First attempt was broken: it flipped position direction hour by hour, creating
enormous artificial turnover (-35 Sharpe, 1330% drawdown). That is not a
control, it is a different strategy.

Correct version flips the direction of each TRADE -- same bars, same entry and
exit times, same turnover, only the side changes.

| sleeve | real | control | GAP |
|---|---|---|---|
| sr | 2.02 | -0.67 | **+2.69** |
| pattern | 1.81 | -0.97 | **+2.78** |

Among the widest gaps measured in this project. The edge is in the direction,
not in the bars selected.

## 4. COSTS — one open concern

**FVG turns over 297x/year**, roughly 25% annual fee drag. It clears that at
215% return, but it is a lot of execution to assume goes perfectly, and it is
the sleeve most exposed to the maker fill rate that has never been measured.

## Changes this audit produced

1. `max_gross` cap on all event sleeves (the FVG leverage bug)
2. BOS dropped -- 0.12 hourly, 0.85 on 63 coins, 1.22 in the blend window.
   Not a reliable sleeve, and it runs at its WORST setting (4h) on the live bot.
3. breakout dropped -- removing it took the book 3.67 -> 3.94 and improved BOTH
   halves. It earned its place when the book was purely dollar-neutral and
   scored 0.25 in the bull; sr (1.96), fvg (3.42) and pattern (2.06) now cover
   that regime. Code kept in breakout.py -- it becomes useful again if the book
   is ever stripped back to cross-sectional only.
4. pattern sleeve ADDED -- 1.92, the most regime-balanced sleeve in the book
   (bull 2.06 / bear 1.87) at only 0.271 gross.
5. cascade rebuilt as an explicit event sleeve rather than an inline ffill.

## Still open

- FVG's 297x turnover vs an unmeasured maker fill rate
- a breakout alternative (the regime-coverage sleeve, if one is still wanted)
- survivorship: the panel holds coins that exist TODAY
