# Full forensic audit

Run against the protocol from the original audit session — the checks that
caught real bugs in the FIRST bot, applied to the code we have been pushing.

## VERDICT

The book at **3.81** survives every check. One new concentration bug found and
fixed (free to fix). Three of the original bot's bugs are confirmed ABSENT.
Two genuine weaknesses remain, both quantified below.

---

## DATA INTEGRITY

**Resampling alignment — PASS.** 6h bars start at 00/06/12/18 UTC, 12h at
00/12. The original bot used fixed 4x1H blocks indexed from the start of the
array, so its 4H bars did not align to real clock boundaries. `resample()` is
time-indexed and does not have this bug.

**Gap stitching — PASS.** Zero gaps >1h across every coin checked. The original
bot's fixed-block approach silently stitched across time discontinuities;
`resample()` cannot.

**Missing-data defaults — PASS.** Zero NaN in close, volume, delta or OI. And
the OI filter is conservative by construction: `(od - od.shift(1)) > 0`
compares FALSE on a NaN, so a missing bar SKIPS the trade rather than taking
it. The original bot defaulted missing delta to 0.0, which distorted z-scores.

**Drawdown measurement — PASS.** 4.40% mark-to-market, worst intraday
excursion -1.83%. The hourly simulator marks EVERY hour including open
positions. The original bot computed drawdown on a realised-PnL cumsum ordered
by entry time, which hides open-position adverse excursion — that is exactly
what a prop firm's daily limit measures, so it mattered.

---

## BUG FOUND: same-coin stacking

`max_gross` capped a sleeve's TOTAL exposure but NOT per coin.

| sleeve | max weight in ONE coin | slots | coin-hours above 1 slot |
|---|---|---|---|
| sr | 0.667 | **4 of 6** | 5,304 |
| fvg | 0.500 | 3 of 6 | **16,525** |
| pattern | 0.500 | 3 of 6 | 3,109 |

Four of S/R's six slots could sit in one coin. The portfolio-level 0.15 cap did
NOT catch it — it applies AFTER dividing by 8 sleeves, so 0.667 becomes 0.083
and passes.

This is the concentration risk the original audit flagged on the old bot:
"a single coin can stack multiple concurrent shorts, concentrating risk across
multiple slots of the 6-position cap."

**Fixing it is free — and helps:**

| sleeve | uncapped | 1-slot cap | maxDD |
|---|---|---|---|
| sr | 1.95 | **1.99** | 21.0 -> 21.8% |
| fvg | 2.96 | 2.96 | **31.7 -> 29.8%** |
| pattern | 1.87 | 1.82 | 22.5 -> 21.5% |
| srflip | 1.69 | **1.79** | **23.1 -> 20.5%** |

Net +0.09 with lower drawdown on three of four. The concentration was
uncompensated risk. `positions_to_hourly` now takes `max_per_coin`.

---

## STATISTICAL VALIDITY

**Multiple testing — PASS, comfortably.**

| | |
|---|---|
| daily Sharpe | 4.07, t-stat **7.72** |
| E[max Sharpe] from 100 noise trials | 1.25 |
| E[max Sharpe] from 300 noise trials | **1.45** |
| E[max Sharpe] from 500 noise trials | 1.53 |

~300 tests run on this project. Pure noise would produce a best-of-300 around
1.45. The book is at 4.07.

**Stability across quarters — PASS.** 2.68 / 4.08 / 4.30 / 4.04. All positive,
no single period carrying it.

**Direction control — PASS.** sr +2.69, pattern +2.78 over a random-direction
control that keeps the same bars, entry times and turnover. Note: the ORIGINAL
bot's control was a sign-flip on P&L, which is near-tautological. Ours flips
the direction of each TRADE, which is the valid form.

---

## WEAKNESS 1: coin concentration (real, not fatal)

| universe | Sharpe | ann% | maxDD |
|---|---|---|---|
| all 24 | 3.81 | 38.4% | 4.4% |
| **minus ZEC** | **3.18** | 29.2% | 4.6% |
| minus 1000PEPE | 3.61 | 35.3% | 4.4% |
| minus NEAR | 3.60 | 35.4% | 4.4% |
| **minus top 5** | **2.49** | 20.7% | 4.5% |

ZEC is 17.3% of gross profit and costs 0.63 to remove. It was also the best
buy-and-hold coin in the panel (1.34 Sharpe, 151% annual), so part of that is
the book capturing one exceptional run rather than repeatable edge.

**Nothing is fatal** — 3.18 without ZEC, 2.49 without the top five. And maxDD
stays ~4.5% in every case, so the RISK profile does not depend on any coin.

---

## WEAKNESS 2: FVG marking ambiguity (unresolved)

FVG reads **2.96 hourly-marked** and **1.09 trade-level**. Both have every fix.
The difference is sizing and marking: continuous marking smooths a sleeve that
holds something ~100% of the time; trade-level books lumpy P&L on exit days.
For a SPARSE sleeve the effect reverses (cascade 0.49 hourly vs 1.17 daily).

Which is right depends on how the live bot sizes FVG — a design decision, not
a fact to discover. **Book without FVG: 3.10.** Treat that as the floor.

---

## NOT CHECKABLE FROM THIS DATA

**Survivorship.** The panel holds 24 coins that exist TODAY. Anything liquid in
2024 that has since delisted was never downloaded. No internal test can see it.
Fixing it needs archived Bybit instrument lists.

**Maker fill rate.** Never measured. FVG turns over ~300x/year, so it is the
sleeve most exposed.
