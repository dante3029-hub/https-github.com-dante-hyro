# The book — 3.68

922 days (2024-02-13 -> 2026-08-22), matched-venue taker data, pinned universes.

## Sleeves

| sleeve | config | Sharpe | bull | bear |
|---|---|---|---|---|
| delta | **60 coins**, z7d, hold 14, N=5 | **1.87** | -0.16 | 2.88 |
| sr | 6h break_res +OI, 3ATR stop, hold 15, no TP | 1.58 | 2.09 | 1.30 |
| relvol | **24 coins**, N=8, 20d baseline, hold 7 | 1.40 | 2.20 | 0.99 |
| fvg | 12h detect +OI, 1ATR stop, hold 10, no TP | 1.33 | 2.65 | 0.57 |
| bos8 | **8h** short-only (NOT the live 4h) | 1.22 | 0.58 | 1.46 |
| skew | 24 coins, **60d lookback**, n=6, hold 45 | 1.19 | 0.93 | 1.31 |
| cascade | market z <= -2.5, hold 3d | 1.17 | 0.77 | 1.38 |
| breakout | 50d high, LONG ONLY, 3ATR, hold 20 | 0.99 | **2.27** | -0.37 |

## Blend

| config | Sharpe | bull | bear | 1st | 2nd |
|---|---|---|---|---|---|
| **all 8, equal weight** | **3.68** | **4.02** | 3.54 | 3.74 | 3.67 |
| 7 (no breakout) | 3.71 | 3.76 | 3.68 | 3.72 | 3.70 |
| 7 (no bos8) | 3.43 | 4.02 | 3.14 | 3.59 | 3.29 |

Breakout costs 0.03 overall and buys **+0.26 in the bull half**. Near-free
insurance against the regime the neutral book cannot capture.

Halves 3.74 / 3.67, bull 4.02 / bear 3.54. Compare the neutral-only book
earlier at 0.25 bull / 2.54 bear -- the regime dependence is largely gone.

## How it got here today

| change | effect |
|---|---|
| matched-venue taker data (BOS could not fire; delta ranked noise) | delta 0.51 -> 1.28 |
| delta hold 5 -> 14 (IC peaks at 21-30d, not 5d) | blend 3.24 -> 3.44, turnover 73x -> 26x |
| exit sweeps: S/R 3ATR, FVG 1ATR, NO take-profits | +0.15 |
| BOS moved off 4h (its worst setting) to 8h | +0.12 |
| long-only breakout sleeve added | bull half +0.26 |
| sleeve-specific universes (delta 60, relvol 24) | +0.29 standalone |
| skew lookback 45d -> 60d, n=6 | 0.88 -> 1.19 |

## Caveats, unchanged

1. **922-day window excludes 2023**, which was a +102% market and hostile to a
   neutral book. Full-history figures are roughly 1.9-2.2.
2. **Halve for live expectation** per the manuals: ~1.8.
3. **~250 tests run.** The noise ceiling is high; treat single cells sceptically
   and prefer the ones with smooth parameter surfaces.
4. **Survivorship is unquantified** -- the panel holds coins that exist today.
5. **The live bot runs BOS at 4h**, which tested 0.33 with a control of 0.74.

## Still open

- S/R, FVG, BOS and breakout are not yet weight matrices, so the netting
  simulator covers only the 4 cross-sectional sleeves
- maker vs taker fill rate, never measured
- vol targeting, CPCV
