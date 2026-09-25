# The verified book — 3.81

Every sleeve audited against its source. Five bugs found and fixed. This is the
state to build from.

## THE BOOK

| sleeve | config | Sharpe | bull | bear | maxDD |
|---|---|---|---|---|---|
| fvg | 12h, +OI, **1.5ATR**, hold 10 | 2.96 | 3.34 | 2.62 | 31.7% |
| sr | 6h break, +OI, 3ATR, hold 15 | 1.95 | 2.05 | 1.85 | 21.0% |
| pattern | 6h MarkitTick bull, +OI, 2ATR, hold 15 | 1.87 | 1.88 | 1.91 | 22.5% |
| srflip | 6h retest, **2ATR**, hold 15 | 1.69 | 1.87 | 1.50 | 23.1% |
| delta | 60 coins, z7d, hold 14, N=5 | 1.41 | 0.77 | 2.04 | 32.9% |
| skew | 24 coins, 60d, n=6, hold 45 | 1.26 | 1.12 | 1.38 | 17.1% |
| relvol | 24 coins, N=8, hold 7 | 1.17 | 1.02 | 1.31 | 14.8% |
| cascade | mkt z <= -2.5, hold 3d | 0.49 | -0.32 | 1.15 | 36.8% |

| book | Sharpe | bull | bear | ann% | maxDD |
|---|---|---|---|---|---|
| **8 sleeves, equal-risk, netted** | **3.81** | 3.43 | 4.18 | 38.4% | **4.4%** |
| 7 (no srflip) | 3.78 | 3.32 | 4.22 | 44.4% | 6.9% |
| **6 (no fvg) — CONSERVATIVE FLOOR** | **3.10** | 2.50 | 3.64 | 37.8% | 8.0% |

Daily Sharpe ~4.0.

## THE FIVE BUGS

**1. Wrong volume series (S/R).** The Pine's `upAndDownVolume()` signs volume by
CANDLE DIRECTION, not taker flow. Using real taker delta is different data:
631 trades instead of 573.

**2. Break detected on close, not low.** `ta.crossover(low, resistanceLevel_1)`
-- the LOW. The whole candle must clear the box. Affected sr and srflip.

Bugs 1+2 together turned **negative second halves positive**: S/R break went
1.16 (halves 2.50 / -0.44) -> 1.70 (2.03 / 1.33). That sign flip is the
strongest evidence the fixes were right.

**3. Stop-placement lookahead, three sleeves.** `stp = c[eb] - stop*A[eb]` while
entering at `o[eb]`. You do not know the close when you enter at the open.
Worth -0.21 on fvg, -0.09 on sr.

**4. Entry bar never checked for the stop.** The exit loop started at `eb+1`, so
a stop hit on the bar you entered was skipped. **This is the big one.**

| FVG stop | before | after |
|---|---|---|
| 0.25ATR | 2.84 | **0.06** |
| 0.5ATR | 1.70 | 0.30 |
| 1.0ATR (was in the book) | 1.05 | 0.73 |
| **1.5ATR** | -- | **1.09** |

The whole "tight stops work for FVG" finding was an artifact. The relationship
REVERSED once the entry bar was checked. S/R barely moved (1.70 -> 1.69)
because its 2-3ATR stops are wide enough that the entry bar rarely contains
them -- a good consistency check.

**5. A crash introduced while fixing #3** (`open` column never unpacked).

## WHAT PASSED

- **FVG detection: exact match to the Pine**, 575 gaps vs 575, identical set
- **MarkitTick: 24/24** constants and conditions vs its Pine
- **Parity**: `live_signal` and `backtest` fire on the same bar in both
  `sleeve_sr.py` and `sleeve_fvg.py`
- **Direction control**: sr +2.69, pattern +2.78 over random-direction

## THE UNRESOLVED ONE

**FVG reads 2.96 hourly-marked and 1.09 trade-level.** Both have every fix. The
difference is position sizing and marking: continuous marking smooths a sleeve
that holds something ~100% of the time; trade-level books lumpy P&L on exit
days. For a SPARSE sleeve the effect reverses (cascade: 0.49 hourly vs 1.17
daily).

Which is right depends on how the live bot sizes FVG -- a design decision, not
a fact to discover. **3.10 (no fvg) is the number to lean on; 3.81 is the
optimistic case.**

## BUILD PATTERN

`sleeve_sr.py` and `sleeve_fvg.py` are the template: ONE `step(P, i, state)`
advancing a state machine one bar at a time using only data up to `i`.
Backtest sweeps it; live calls it once. Parity is structural, not tested for.

Before this, three implementations of S/R gave 403, 572 and 631 trades. That
cannot happen with one code path.

`SRState` and `FVGState` are JSON-serialisable so the live bot survives
restarts -- srflip must remember "resistance at X broke at bar N" for 20 bars.

## STILL TO PORT THIS WAY

delta, relvol, skew, cascade (cross-sectional -- simpler, no state),
pattern (detector already verified 24/24), srflip (inside sleeve_sr.py).
