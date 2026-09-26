# Candidate sleeves — tested, not adopted

## The structural problem

**Five of the eight sleeves are "long-only price structure on 6-12h bars."**
sr, srflip, pattern, fvg and anything like them all answer the same question:
*is price about to continue up from here?* They correlate 0.4-0.6 with each
other by construction. Any new sleeve of that shape duplicates rather than adds.

Two candidates failed exactly this way:

| candidate | best | corr to sr | in book | why rejected |
|---|---|---|---|---|
| breakout (20/50/100d high) | 1.71 | **0.56** | 3.81 -> 3.77 | same levels, same timing as sr |
| EMA retest (50/100/200) | 1.46 | **0.55** | 3.81 -> 3.60 | srflip with an EMA instead of a volume level |

### EMA retest — the hold requirement does nothing

The idea was to require N consecutive closes above the EMA to confirm the
retest held. Tested across three timeframes:

| 24h ema100 | Sharpe |
|---|---|
| hold 1 | 1.41 |
| hold 2 | 1.45 |
| hold 3 | 1.46 |
| hold 5 | 1.39 |
| hold 8 | **0.93** |

Flat from 1 to 5. Only hold 8 matters, and it makes things worse -- by then the
retest is over and you are chasing. Sound in principle, not paid for.

Best config overall was 24h ema100 (1.45, bull 2.05, bear 0.78) but every
variant had a 52-86% drawdown vs 21-31% for the existing event sleeves.

---

## SPREAD MOMENTUM — the one genuinely orthogonal mechanism

Pairs mean-reversion FAILED (-0.99 to -1.79 across four configs) but was
uncorrelated (-0.04 to -0.09). Since the spread loses money converging, it
trends -- so trade WITH the divergence.

| entry/exit z | Sharpe | bull | bear | maxDD | n |
|---|---|---|---|---|---|
| **1.5 / 0.5** | **0.96** | 0.53 | 1.28 | **11.1%** | 710 |
| 2.0 / 0.5 | 0.82 | 0.22 | 1.25 | 14.6% | 556 |
| 2.5 / 0.5 | 0.35 | -0.02 | 0.62 | 15.4% | 436 |

**11.1% maxDD is the lowest of any sleeve tested** -- less than half relvol's
14.8%. And the correlations are the lowest in the book:

| vs | corr | vs | corr |
|---|---|---|---|
| delta | 0.06 | sr | 0.09 |
| relvol | 0.17 | pattern | 0.07 |
| skew | -0.06 | fvg | 0.06 |
| cascade | 0.03 | srflip | 0.05 |

### In the book

| book | Sharpe | ann% | maxDD |
|---|---|---|---|
| 8 (current) | 3.81 | 38.4% | 4.4% |
| 9 (+ spread mom) | 3.79 | 30.8% | **3.3%** |

Sharpe flat, maxDD down 25%, annual return down 7.6 points. **Not adopted** --
at the same Sharpe it is a lateral move, and the same drawdown reduction is
available by simply sizing down, which costs nothing and adds no code.

**Keep it as the lever.** If the universe widens and the event sleeves dilute,
or if live drawdown creeps above the backtest, this is what to reach for.

Only 10 pairs cleared a 0.7 trailing correlation on the formation window
(XLM/XRP 0.87, AVAX/SOL 0.77, ADA/DOT 0.77). Crypto is correlated to BTC but
not stably correlated pairwise.

---

## HOUR OF DAY — a trap

The raw effects look compelling:

| UTC hour | mean bp | Sharpe |
|---|---|---|
| 22:00 | +5.36 | 6.79 |
| 21:00 | +4.43 | 4.94 |
| 13:00 | -4.45 | -4.86 |

**The tradeable version scores -10.37.** Trading in and out eight times a day at
taker fees destroys it. This is the candidate the v15 spec listed at "+1.41
@maker" -- it needs a fill at a fixed hour, the hardest possible case for a
post-only order, and that fill rate has never been measured.
