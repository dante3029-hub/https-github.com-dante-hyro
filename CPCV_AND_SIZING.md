# CPCV on the current book, and sizing against a static floor

## CPCV — 163 partitions

The spec's CPCV (3.10 +/- 1.13) was run on the OLD book -- core, pattern, w5 --
with the corrupt taker column. It validated nothing built since.

| K, test groups | partitions | mean | median | min | >1.0 | negative |
|---|---|---|---|---|---|---|
| K=6, test=2 | 15 | **3.53** | 3.51 | 1.65 | 100% | **0%** |
| K=8, test=2 | 28 | 3.31 | 3.31 | -0.28 | 89% | 7% |
| K=10, test=3 | 120 | **3.41** | 3.54 | 0.41 | 97% | **0%** |

Full-sample 3.68 sits ABOVE the CPCV mean, which is the right direction -- the
headline is not inflated relative to sub-periods. Two of 163 partitions go
negative, both in the K=8 split.

### Why the blend exists

| sleeve | CPCV mean | min | negative partitions |
|---|---|---|---|
| delta | 1.85 | -1.07 | 14% |
| sr | 1.52 | -0.44 | 7% |
| relvol | 1.34 | -1.60 | 7% |
| fvg | 1.27 | -1.60 | 18% |
| bos8 | 1.12 | -1.25 | 21% |
| cascade | 1.11 | 0.12 | 0% |
| skew | 1.09 | -0.33 | 11% |
| **breakout** | **0.40** | -2.39 | **36%** |
| **BLEND** | **3.31** | -0.28 | **7%** |

Every sleeve loses money in some partitions -- breakout in over a third. The
blend loses in 0-7%. Breakout being weakest here is expected: it is
regime-dependent by design, so bear-stretch partitions show it negative. It
still earns its place by covering the bull half (+0.26).

## Sizing -- STATIC floor, fresh $200k

Floor $180,000, static. **Because it is static the buffer GROWS as you profit**
-- an early loss is far more dangerous than a late one, which is why a single
"max drawdown" figure is misleading and the path has to be simulated.

Target +$20k. 120 trading days. 20,000 paths.

| daily vol | Sharpe 1.8 | Sharpe 2.5 | Sharpe 3.68 |
|---|---|---|---|
| | fail / pass | fail / pass | fail / pass |
| $1,000 | 0.6 / 28.7 | 0.2 / 43.9 | 0.0 / 68.5 |
| **$2,000** | **9.2 / 68.9** | 4.9 / 80.6 | 1.5 / 93.3 |
| $3,000 | 19.2 / 76.1 | 12.6 / 84.6 | 5.6 / 93.6 |
| $4,000 | 25.5 / 73.8 | 18.6 / 81.0 | 10.5 / 89.4 |
| $7,000 | 34.2 / 65.8 | 28.9 / 71.1 | 21.1 / 78.9 |

**Recommendation: $2,000/day at the honest Sharpe of 1.8** -- 68.9% pass,
9.2% fail. That is ~1% of equity in daily vol.

Pass rate PEAKS and then falls: 76.1% at $3,000 down to 65.8% at $7,000. More
size does not help past a point -- you fail before you can compound.

### Note on the earlier $7,000/day recommendation

That was made on a book measured with the loss-erasure bug and a delta sleeve
ranking a corrupt column. At 34% failure it was far too aggressive.

### On the CURRENT account ($183,800, buffer $3,800)

There is no good sizing. At Sharpe 1.8 and $500/day: 3.7% pass, 16.6% fail.
You need +$16,200 with $3,800 of room -- a 4.3:1 requirement on the account
before any strategy consideration.
