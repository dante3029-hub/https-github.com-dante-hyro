# Book v16 — rebuilt on matched-venue data

Everything here is recomputed after finding that `clean_panel`'s taker column
was another venue's volume: taker buy exceeded TOTAL volume on 60% of bars
(median ratio 1.07, stdev 0.30). Any "delta" derived from it was meaningless.

## What that broke

| affected | consequence |
|---|---|
| **BOS sleeve** | needed `dn < 0`, which was NEVER true -> zero weights on 300+ live cycles |
| **delta sleeve** | ranked on noise; measured 0.51, actually 1.28 on clean data |
| **v15 "core" at 1.88** | delta + BOS weighted 70/30, both compromised |
| **the IC table** | 1d IC was 0.0276 (t=3.24) on bad data, 0.0050 (t=0.75) on clean |

Fixed by fetching Binance futures klines, where volume and taker_buy come from
the SAME response and cannot drift apart (`fetch_taker.py`). Sanity check on
29,371 SOL bars: 0 with taker > volume, 58% with negative delta.

## Signal half-life — the spec's conclusion HOLDS, and is stronger

| horizon | clean IC | t-stat | spec (corrupt) |
|---|---|---|---|
| 1d | 0.0050 | **0.75** | 0.0276 (t 3.24) |
| 3d | 0.0275 | 4.11 | 0.0527 |
| 7d | 0.0359 | 5.42 | 0.0463 |
| 14d | 0.0650 | 9.98 | 0.0640 |
| 21d | **0.0793** | **12.05** | 0.0678 |
| 30d | 0.0835 | 12.83 | — |

The edge strengthens monotonically to 30 days. The 1-day signal was an artifact
of the corrupt column. hold=5 (found by grid search -- exactly what the manual
warns against) uses the weakest part of the signal.

## The book

| sleeve | config | Sharpe |
|---|---|---|
| sr | 6h break_res +OI, stop 3ATR, hold 15, no TP | 1.58 |
| relvol | N=8, 20d baseline | 1.40 |
| fvg | 12h detect +OI, stop 1ATR, hold 10, no TP | 1.33 |
| delta | z7d, **hold 14**, N=5 | 1.28 |
| cascade | market z <= -2.5, hold 3d | 1.17 |
| skew | 45d realised, hold 45 | 0.89 |
| bos8 | 8h short-only (NOT 4h -- see below) | 0.88 |
| **BLEND, equal weight** | | **3.44** (halves 3.69 / 3.18) |

## Progression today

| change | blend |
|---|---|
| starting point | 2.99 |
| exit sweeps (S/R 3ATR, FVG 1ATR, no TPs) | 3.14 |
| + bos8 | 3.24 |
| delta hold 5 -> 14 | **3.44** |

## Things that did NOT make it

| candidate | best | why not |
|---|---|---|
| divergence | 0.49 | 20+ configs, one marginal survivor; costs 0.13 in blend |
| Elliott wave | — | 16 configs, controls beat it everywhere |
| funding z-fade | 0.09 | 12 configs, all negative |
| Ichimoku | -0.16 | all 7 signals negative |
| strat8 | 0.75 | below blend average; neutral at half weight |
| BOS at 4h (LIVE SETTING) | 0.33 | control scored 0.74 -- random beats it |

## Open, and load-bearing

1. **BOS runs at 4h live.** That is its worst setting. 8h or 12h or off.
2. **Turnover fell from 73x to 26x** with hold=14, so the unmeasured maker fill
   rate matters far less than it did. Still worth measuring.
3. **No portfolio simulator.** Sleeves are summed as independent streams. Real
   netting would cut fees; real caps would cut exposure. 3.44 is an upper bound.
4. **Halve for live expectation** per the manuals: ~1.7.
