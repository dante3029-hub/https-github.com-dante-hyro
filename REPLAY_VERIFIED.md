# Full system replay — the live code path, verified

## Result

**24 coins, 2 years (2024-06 to 2026-06), 244,764 bar-steps, 1,930 opens.**

| integrity check | result |
|---|---|
| stops on the correct side | **1930/1930** |
| duplicate opens while already in a trade | **0** |
| max slots <= 6 for every sleeve | **True** |
| opens - closes = still open | **11 vs 11** (books balance) |
| mid-run restart | **serialise -> discard -> rebuild, no position lost or duplicated** |

| sleeve | opens | max concurrent |
|---|---|---|
| fvg | 1070 | 6 (hit the cap) |
| pattern | 482 | 1 |
| sr | 232 | 3 |
| srflip | 146 | 4 |

Exits: **38% stop, 62% time** -- matching the design (wide stops, fixed hold).

## The trade log reads correctly

```
2024-06-08 00:00  OPEN  srflip  1000RATS  +1  px 0.1495  stop 0.1252
2024-06-11 00:00  CLOSE srflip  1000RATS      px 0.1252  (stop)
2024-12-10 00:00  OPEN  fvg     1000RATS  -1  px 0.0972  stop 0.1151
2024-12-14 12:00  CLOSE fvg     1000RATS      px 0.0858  (time)
```

Entry at the bar AFTER the signal. Stop the right distance on the right side.
Stopped out at exactly the stop price.

## fast_detect.py — what made this runnable

The detectors recompute from scratch on every call: `sr2.signals()` builds
levels across the whole series, srflip loops every bar, the pattern detector
rescans all 16 patterns. O(n) per call, so bar-by-bar replay is O(n^2) -- a
six-month replay across ten coins **did not finish**.

Precomputing per coin and indexing makes `fired(i)` an array read:

| | |
|---|---|
| build once per coin | 0.09s |
| 80 bars, slow path | 2.73s |
| 80 bars, fast path | 0.00002s |
| **speedup** | **156,935x** |
| a 4,000-bar replay | 2 minutes -> **0.1s** |

**Verified identical on bars that actually fire**, not on an empty window:

| sleeve | fires | controls | agree |
|---|---|---|---|
| sr | 60 | 60 | **120/120** |
| srflip | 12 | 12 | **24/24** |
| fvg | 428 | 43 | **471/471** |
| pattern | 66 | 66 | **132/132** |

The first verification attempt checked a window with ZERO signals in both
paths and reported "identical". That proves nothing -- redone on every firing
bar plus matched controls.

## Why this matters beyond speed

The replay is the only test that exercises the live path end to end. If it
takes two minutes it gets run once; at 0.1s it gets run before every change
and every day of paper trading.

## What it catches that a backtest cannot

- a tracker opening a slot it should not, or failing to close one
- state that does not survive a restart mid-run
- cadence bugs
- duplicate positions in one coin
- stops on the wrong side of entry

All five now have a passing test.

## Still not proven

**Nothing here has touched a live API.** Order rejection, exchange downtime,
the perp not being tradeable in the size wanted, and the kill switch firing on
real intraday equity are all unproven. That is what the paper month is for.
